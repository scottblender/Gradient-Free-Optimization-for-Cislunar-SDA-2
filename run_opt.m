% ---- run_opt.m ---- %
clear; close all; clc;

% ---------------- Results output ----------------
MAKE_PLOTS = strcmp(getenv("MAKE_PLOTS"),"1"); % Display only; never saved.
STUDY_ID = string(getenv("STUDY_ID"));
if strlength(STUDY_ID) == 0, STUDY_ID = "manual_fe_run"; end

% ---------------- Figure defaults ----------------
set(groot, ...
    'defaultAxesFontSize',16, ...
    'defaultAxesFontWeight','bold', ...
    'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontSize',12, ...
    'defaultTextFontWeight','bold', ...
    'defaultTextFontName','Times New Roman', ...
    'defaultLegendFontSize',12, ...
    'defaultLegendFontWeight','bold', ...
    'defaultAxesLabelFontSizeMultiplier',1.0, ...
    'defaultAxesTitleFontSizeMultiplier',1.0, ...
    'defaultLineLineWidth',1.8);

% ---------------- Load JPL data ----------------
thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);

% Resolve code, input data, and output paths for local and batch runs.
addpath(thisDir);
projectPaths = setup_project();

catalogPath = projectPaths.catalog;
S = load(catalogPath);
CatalogDir = projectPaths.data;
T1 = S.T;
catalogHash = study_hash(catalogPath,"file");
t_lg = S.t_lg;
s_lg = S.s_lg;

% ---------------- Optimizer inputs ----------------
% Supported SOO methods: GA, PSO, BAYESIAN, ABC, and ACO.
OPTIMIZER_MODE = 'GA';

envMode = getenv("OPTIMIZER_MODE");
if ~isempty(envMode)
    OPTIMIZER_MODE = envMode;
end
OPTIMIZER_MODE = upper(string(OPTIMIZER_MODE));

% Stopping Criteria - SOO comparisons use a universal FE budget.
FE_BUDGET = 6000;

v = getenv("MAX_EVALS");
if ~isempty(v)
    FE_BUDGET = str2double(v);
end

validateattributes(FE_BUDGET, {'numeric'}, ...
    {'scalar','real','finite','integer','positive'});

% All supported optimizers stop on the same objective-function-evaluation budget.
supportedOptimizers = ["GA","PSO","BAYESIAN","ABC","ACO"];
assert(ismember(OPTIMIZER_MODE,supportedOptimizers), ...
    "Unknown OPTIMIZER_MODE: %s",OPTIMIZER_MODE);

USE_PARALLEL = true;
v = getenv("USE_PARALLEL_OPT");
if ~isempty(v)
    USE_PARALLEL = (str2double(v) ~= 0);
end

% ---------------- JPL constants ----------------
mu = 1.215058560962404E-2;
LU = 384400;     % km
TU = 375695;     % seconds
VU = LU / TU;    % km/s

ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% ---------------- Lunar Gateway ICs ----------------
s_lg_ic     = [1.02202108343387, 0, -0.182096487798513, 0, -0.103255420206012, 0]';
tspan_lg_ic = [0, 1.51110546287394];

% ---------------- Catalog / orbit database ----------------
num_orbits      = height(T1);
slots_per_orbit = 50;

tf          = T1.("Period (TU) ");
states      = T1.("state");
times       = T1.("time");
stabilities = T1.("Stability index  ");

OrbitCacheDir = projectPaths.orbitCache;
if ~exist(OrbitCacheDir,'dir'), mkdir(OrbitCacheDir); end

catalogFile = "JPL_CR3BP_OrbitCatalog.mat";
orbitDbCacheFile = fullfile(OrbitCacheDir, ...
    sprintf('orbit_database_slots_%d.mat', slots_per_orbit));

rebuildOrbitDb = true;

if isfile(orbitDbCacheFile)
    try
        C = load(orbitDbCacheFile, 'orbit_database', 'cacheMeta');
        if isfield(C,'orbit_database') && numel(C.orbit_database) == num_orbits && ...
                isfield(C,'cacheMeta') && isfield(C.cacheMeta,'catalogHash') && ...
                string(C.cacheMeta.catalogHash) == catalogHash && ...
                isfield(C.cacheMeta,'slotDefinition') && ...
                string(C.cacheMeta.slotDefinition) == "equal_time_no_endpoint_v1"
            orbit_database = C.orbit_database;
            rebuildOrbitDb = false;
            safe_printf('Loaded cached orbit database from:\n  %s\n', orbitDbCacheFile);
        end
    catch ME
        safe_printf(2, 'WARNING: failed to load orbit database cache: %s\n', ME.message);
        rebuildOrbitDb = true;
    end
end

if rebuildOrbitDb
    safe_printf('Building orbit database for %d slots/orbit...\n', slots_per_orbit);

    orbit_database = cell(num_orbits, 1);

    parfor i = 1:num_orbits
        t_raw  = times{i};
        s_raw  = states{i};
        period = tf(i);

        t_slots = (0:slots_per_orbit-1)' * period / slots_per_orbit;

        [t_unique, idx_u] = unique(t_raw);
        s_unique = s_raw(idx_u, :);
        F        = griddedInterpolant(t_unique, s_unique, 'spline');
        s_slots  = F(t_slots);

        orbit_database{i} = s_slots;
    end

    cacheMeta = struct();
    cacheMeta.created         = string(datetime('now'));
    cacheMeta.catalogFile     = string(catalogFile);
    cacheMeta.num_orbits      = num_orbits;
    cacheMeta.catalogHash = catalogHash;
    cacheMeta.slotDefinition = "equal_time_no_endpoint_v1";
    cacheMeta.slots_per_orbit = slots_per_orbit;

    try
        save(orbitDbCacheFile, 'orbit_database', 'cacheMeta', '-v7.3');
        safe_printf('Saved orbit database cache to:\n  %s\n', orbitDbCacheFile);
    catch ME
        safe_printf(2, 'WARNING: failed to save orbit database cache: %s\n', ME.message);
    end
end

% ---------------- Run directory / outputs ----------------
runDirEnv = getenv("RUN_DIR");
RunDir = "";

if ~isempty(runDirEnv)
    if ~exist(runDirEnv,'dir')
        mkdir(runDirEnv);
    end
    cd(runDirEnv);
    RunDir = pwd;
end

% ---------------- Visibility parameters ----------------
% Absolute minimum LOS separations measured from each body center.
% calc_visibility combines exclusion and occultation using max(...).
sun_exclusion_deg   = 20;
moon_exclusion_deg  = 10;
earth_exclusion_deg = 15;     
sun_exclusion   = deg2rad(sun_exclusion_deg);
moon_exclusion  = deg2rad(moon_exclusion_deg);
earth_exclusion = deg2rad(earth_exclusion_deg);

theta0 = 0;
i_sun  = deg2rad(0);

sunFcn = @(t) sun_pos_bc4bp(t, LU, TU, theta0, i_sun);

useScreening = true;
costFlags = struct('J1', true, 'J2', true, 'J3', true);

v = getenv("USE_SCREENING");
if ~isempty(v), useScreening = (str2double(v) ~= 0); end

vj1 = getenv("USE_J1"); if ~isempty(vj1), costFlags.J1 = (str2double(vj1) ~= 0); end
vj2 = getenv("USE_J2"); if ~isempty(vj2), costFlags.J2 = (str2double(vj2) ~= 0); end
vj3 = getenv("USE_J3"); if ~isempty(vj3), costFlags.J3 = (str2double(vj3) ~= 0); end

% ---------------- Measurement model ----------------
measCfg = struct();
measCfg.type = "ANGLES_ONLY";

v = getenv("MEAS_MODEL");
if ~isempty(v)
    measCfg.type = upper(string(v));
end
if measCfg.type ~= "ANGLES_ONLY" && measCfg.type ~= "ANGLES_RANGE"
    error("Unknown MEAS_MODEL: %s", measCfg.type);
end

switch measCfg.type
    case "ANGLES_ONLY"
        measCode = "AO";
    case "ANGLES_RANGE"
        measCode = "AR";
    otherwise
        measCode = "UNK";
end

% ---------------- Objective callback ----------------
% FE histories are recorded by the optimizer-specific callbacks below.
dq = [];

opt_flag          = 'SOO';
const_stabilities = parallel.pool.Constant(stabilities);
const_orbit_db    = parallel.pool.Constant(orbit_database);

% ---------------- Mission type ----------------
MISSION_TYPE = "LUNAR_GATEWAY";

envMission = getenv("MISSION_TYPE");
if ~isempty(envMission)
    MISSION_TYPE = upper(string(envMission));
end

% ---------------- Mission config ----------------
missionCfg = struct();
missionCfg.type = upper(string(MISSION_TYPE));

missionCfg.optimization.numObservers = 3;
v = getenv("NUM_OBSERVERS");
if ~isempty(v)
    missionCfg.optimization.numObservers = str2double(v);
end
if isnan(missionCfg.optimization.numObservers) || missionCfg.optimization.numObservers < 1
    missionCfg.optimization.numObservers = 3;
end
missionCfg.optimization.numObservers = round(missionCfg.optimization.numObservers);

switch missionCfg.type

    case "LUNAR_GATEWAY"
        fixedCase = target_case_config("LUNAR_GATEWAY");
        missionCfg.gateway = fixedCase.gateway;

    case "PERIODIC_ORBIT"
        missionCfg.periodic.orbitIndex = 1;
        missionCfg.periodic.dt         = 0.001;
        missionCfg.periodic.Nperiods   = 1;

    case "GATEWAY_IMPULSE"
        fixedCase = target_case_config("GATEWAY_IMPULSE");
        missionCfg.impulse = fixedCase.impulse;

        v = getenv("IMPULSE_DURATION_TU");
        if ~isempty(v)
            missionCfg.impulse.duration_TU = str2double(v);
        end

        v = getenv("IMPULSE_DV_MPS");
        if ~isempty(v)
            missionCfg.impulse.deltaV_m_s = str2double(v);
        end

        v = getenv("IMPULSE_DIRECTION");
        if ~isempty(v)
            missionCfg.impulse.direction = upper(string(v));
        end

        validateattributes(missionCfg.impulse.duration_TU, {'numeric'}, ...
            {'scalar','real','finite','positive'});
        validateattributes(missionCfg.impulse.deltaV_m_s, {'numeric'}, ...
            {'scalar','real','finite','positive'});

        missionCfg.impulse.deltaV_LU_TU = ...
            (missionCfg.impulse.deltaV_m_s / 1000) / VU;

    case "LOW_THRUST_TRANSFER"
        fixedCase = target_case_config("LOW_THRUST_TRANSFER");
        missionCfg.transfer = fixedCase.transfer;

    otherwise
        error("Unknown MISSION_TYPE: %s", missionCfg.type);
end

% ---------------- Override number of periods ----------------
v = getenv("NPERIODS");
if ~isempty(v)
    nper = str2double(v);
    if ~isnan(nper) && nper > 0
        nper = round(nper);
        switch missionCfg.type
            case "LUNAR_GATEWAY"
                missionCfg.gateway.Nperiods = nper;
            case "PERIODIC_ORBIT"
                missionCfg.periodic.Nperiods = nper;
        end
    end
end

% ---------------- Run tag / logging / outputs ----------------
vseed = getenv("SEED");
if isempty(vseed), vseed = "0"; end

seedVal = str2double(vseed);
if isnan(seedVal), seedVal = 0; end
rng(seedVal, 'twister');

% Keep measurement noise separate from the optimizer seed.
measCfg.noiseSeed = 1001;

v = getenv("MEAS_NOISE_SEED");
if ~isempty(v)
    measCfg.noiseSeed = str2double(v);
end

validateattributes(measCfg.noiseSeed, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', '>=', 0, '<=', 2^32-1});

RUN_TAG = sprintf('%s_scr%d_%s_J%d%d%d_seed%03d', char(OPTIMIZER_MODE), ...
    double(useScreening), char(measCode), ...
    double(costFlags.J1), double(costFlags.J2), double(costFlags.J3), seedVal);

if strlength(string(RunDir)) == 0
    ts = string(datetime('now','Format','yyyyMMdd_HHmmss'));

    switch upper(string(missionCfg.type))
        case "LOW_THRUST_TRANSFER"
            missionCode = "lt";
        case "LUNAR_GATEWAY"
            missionCode = "lg";
        case "GATEWAY_IMPULSE"
            missionCode = "gi";
        case "PERIODIC_ORBIT"
            missionCode = "po";
        otherwise
            missionCode = lower(string(missionCfg.type));
    end

    num_obs_local = missionCfg.optimization.numObservers;

    if missionCfg.type == "LUNAR_GATEWAY"
        nper_local = missionCfg.gateway.Nperiods;
        caseName = sprintf('run_%s_o%d_p%d', char(RUN_TAG), num_obs_local, nper_local);
    elseif missionCfg.type == "PERIODIC_ORBIT"
        nper_local = missionCfg.periodic.Nperiods;
        caseName = sprintf('run_%s_o%d_p%d', char(RUN_TAG), num_obs_local, nper_local);
    else
        caseName = sprintf('run_%s_o%d', char(RUN_TAG), num_obs_local);
    end

    RunsRoot   = projectPaths.runs;
    BatchDir   = fullfile(RunsRoot, ts);
    MissionDir = fullfile(BatchDir, missionCode);
    RunDir     = fullfile(MissionDir, caseName);

    if ~exist(RunsRoot,'dir'), mkdir(RunsRoot); end
    if ~exist(BatchDir,'dir'), mkdir(BatchDir); end
    if ~exist(MissionDir,'dir'), mkdir(MissionDir); end
    if ~exist(RunDir,'dir'), mkdir(RunDir); end

    cd(RunDir);
end

DataDir = fullfile(RunDir, "data");
LogDir  = fullfile(RunDir, "logs");

TransferCacheDir = projectPaths.transferCache;

if ~exist(DataDir,'dir'), mkdir(DataDir); end
assert(~isfile(fullfile(DataDir,'optimization_run.mat')) && ...
    ~isfile(fullfile(DataDir,'tracking_data.mat')), ...
    'Run output already exists. Choose a new RUN_DIR; do not overwrite runs.');
if ~exist(LogDir,'dir'), mkdir(LogDir); end
if ~exist(TransferCacheDir,'dir'), mkdir(TransferCacheDir); end

switch upper(string(missionCfg.type))
    case "LOW_THRUST_TRANSFER"
        missionCodeShort = "lt";
    case "LUNAR_GATEWAY"
        missionCodeShort = "lg";
    case "GATEWAY_IMPULSE"
        missionCodeShort = "gi";
    case "PERIODIC_ORBIT"
        missionCodeShort = "po";
    otherwise
        missionCodeShort = lower(string(missionCfg.type));
end

FILE_TAG = sprintf('%s_%s_o%d', char(RUN_TAG), char(missionCodeShort), missionCfg.optimization.numObservers);

% ---------------- Output / log files ----------------
setenv('SAFE_FALLBACK_FILE', fullfile(LogDir, sprintf('safe_output_fallback_%s.txt', FILE_TAG)));

try
    diaryFile = fullfile(LogDir, sprintf('matlab_diary_%s.txt', FILE_TAG));
    diary(diaryFile);
    diary on
catch
end

safe_printf('RUN START: %s\n', string(datetime('now')));
safe_printf('Run directory: %s\n', string(RunDir));
drawnow;

% ---------------- EKF parameters ----------------
if strcmp(MISSION_TYPE, "LOW_THRUST_TRANSFER")
    pos_var  = (1 / LU)^2;
    vel_var  = (10 / (VU * 1000))^2;
    P_0 = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);

    q_pos = 6.25e-4;
    q_vel = 6.25e-4;
    Q_k = diag([q_pos q_pos q_pos q_vel q_vel q_vel]);

    r_ang   = 1e-8;
    r_range = (1 / LU)^2;

    switch measCfg.type
        case "ANGLES_ONLY"
            R_k = diag([r_ang r_ang]);
        case "ANGLES_RANGE"
            R_k = diag([r_ang r_ang r_range]);
    end

elseif strcmp(MISSION_TYPE, "LUNAR_GATEWAY") || ...
        strcmp(MISSION_TYPE, "GATEWAY_IMPULSE") || ...
        strcmp(MISSION_TYPE, "PERIODIC_ORBIT")
    pos_var  = (1 / LU)^2;
    vel_var  = (10 / (VU * 1000))^2;
    P_0 = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);

    q_pos = 1e-8;
    q_vel = 1e-8;
    Q_k = diag([q_pos q_pos q_pos q_vel q_vel q_vel]);

    r_ang   = 1e-8;
    r_range = (1 / LU)^2;

    switch measCfg.type
        case "ANGLES_ONLY"
            R_k = diag([r_ang r_ang]);
        case "ANGLES_RANGE"
            R_k = diag([r_ang r_ang r_range]);
    end
end

% ---------------- Cost thresholds ----------------
costCfg = struct();
costCfg.weights = [1, 1, 0.1];

switch upper(string(missionCfg.type))
    case "LOW_THRUST_TRANSFER"
        costCfg.pos_rmse_acc = 100 / LU;
        costCfg.vel_rmse_acc = 0.1 / VU;
        costCfg.sigma_pos_acc = 100 / LU;
        costCfg.sigma_vel_acc = 0.1 / VU;
        costCfg.stability_acc = 1.0;

    case {"LUNAR_GATEWAY", "GATEWAY_IMPULSE"}
        costCfg.pos_rmse_acc = 1 / LU;
        costCfg.vel_rmse_acc = 1.0e-3 / VU;
        costCfg.sigma_pos_acc = 1 / LU;
        costCfg.sigma_vel_acc = 1.0e-3 / VU;
        costCfg.stability_acc = 1.0;

    case "PERIODIC_ORBIT"
        costCfg.pos_rmse_acc = 0.1 / LU;
        costCfg.vel_rmse_acc = 1.0e-4 / VU;
        costCfg.sigma_pos_acc = 0.1 / LU;
        costCfg.sigma_vel_acc = 1.0e-4 / VU;
        costCfg.stability_acc = 1.0;

    otherwise
        costCfg.pos_rmse_acc = 1 / LU;
        costCfg.vel_rmse_acc = 1.0e-3 / VU;
        costCfg.sigma_pos_acc = 1 / LU;
        costCfg.sigma_vel_acc = 1.0e-3 / VU;
        costCfg.stability_acc = 1.0;
end

% ---------------- Dynamic optimizer sizing ----------------
num_obs_cfg = missionCfg.optimization.numObservers;
nVars_common = 2 * num_obs_cfg;
LB_common = repmat([1, 1], 1, num_obs_cfg);
UB_common = repmat([num_orbits, slots_per_orbit], 1, num_obs_cfg);

% ---------------- Build/load target truth ----------------
useTransferCache = true;

if missionCfg.type == "LOW_THRUST_TRANSFER" && useTransferCache
    cacheKey = make_transfer_cache_key(missionCfg);
    cacheKey = cacheKey + "_" + study_hash({missionCfg.transfer,mu});
    cacheFile = fullfile(TransferCacheDir, cacheKey + ".mat");

    loadedFromCache = false;

    if isfile(cacheFile)
        try
            safe_printf('Loading cached transfer truth from:\n  %s\n', cacheFile);
            C = load(cacheFile, 't_target', 's_target', 'truthInfo', 'cacheMeta');
            t_target  = C.t_target;
            s_target  = C.s_target;
            truthInfo = C.truthInfo;
            if isfield(C, 'cacheMeta')
                safe_printf('Cached transfer key: %s\n', string(C.cacheMeta.cacheKey));
            end
            loadedFromCache = true;
        catch ME
            safe_printf(2, 'WARNING: failed to load transfer cache, rebuilding: %s\n', ME.message);
            try
                delete(cacheFile);
            catch
            end
        end
    end

    if ~loadedFromCache
        safe_printf('No valid cached transfer found. Computing transfer truth.\n');
        [t_target, s_target, truthInfo] = build_target_truth( ...
            missionCfg, T1, orbit_database, times, states, mu, ode_opts);

        cacheMeta = struct();
        cacheMeta.cacheKey    = cacheKey;
        cacheMeta.missionType = string(missionCfg.type);
        cacheMeta.created     = string(datetime('now'));
        cacheMeta.mu          = mu;
        tmpFile = fullfile(TransferCacheDir, cacheKey + "_t.mat");

        try
            if isfile(tmpFile), delete(tmpFile); end
            if isfile(cacheFile), delete(cacheFile); end
            save(tmpFile, 't_target', 's_target', 'truthInfo', 'cacheMeta', '-v7.3');
            movefile(tmpFile, cacheFile, 'f');
            safe_printf('Saved transfer truth cache to:\n  %s\n', cacheFile);
        catch ME
            safe_printf(2, 'WARNING: failed to save transfer cache: %s\n', ME.message);
            try
                if isfile(tmpFile), delete(tmpFile); end
            catch
            end
        end
    end
else
    [t_target, s_target, truthInfo] = build_target_truth( ...
        missionCfg, T1, orbit_database, times, states, mu, ode_opts);
end

% ---------------- Moon impact check ----------------
if contains(string(missionCfg.type), "TRANSFER") || ...
        missionCfg.type == "GATEWAY_IMPULSE"
    r_moon = [1 - mu, 0, 0];
    R_moon = 1737.1 / LU;   % LU
    h_min  = 100 / LU;    % example 100 km keep-out altitude

    d_moon = vecnorm(s_target(:,1:3) - r_moon, 2, 2);
    min_d_moon = min(d_moon);

    if min_d_moon <= (R_moon + h_min)
        error('Target trajectory violates Moon keep-out zone. Min distance = %.6e LU.', min_d_moon);
    end
end

% ---------------- EKF cadence ----------------
EKF_DT = 0.01;
v = getenv("EKF_DT");
if ~isempty(v), EKF_DT = str2double(v); end

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
% Settings and the time grid are saved in optimization_run.mat/tracking_data.mat.

% ---------------- Objective function wrapper ----------------
RawObjFcn = @(x) objective_wrapper( ...
    x, const_orbit_db, const_stabilities, s_target_ekf, ...
    t_target_ekf, P_0, Q_k, R_k, mu, LU, ...
    sunFcn, sun_exclusion, moon_exclusion, earth_exclusion, ...
    opt_flag, OPTIMIZER_MODE, dq, useScreening, ...
    costFlags, costCfg, measCfg);

ObjFcn = RawObjFcn;

% Start the pool before timing so pool startup is not charged to a
% particular optimizer. All five SOO methods use parallel objective
% evaluation when USE_PARALLEL is true.
if USE_PARALLEL
    p = gcp('nocreate');
    if isempty(p)
        parpool;
    end
end

reset_fe_history();
history = table();
actualEvals = NaN;
solverFunccount = NaN;
parallelOverflowEvals = 0;
budgetRuntime_s = NaN;

% ---------------- Reproducibility metadata ----------------
solverSettingsText = "";
objectiveErrorCount = 0;
x_best = [];
min_cost = Inf;

codeFiles = ["run_opt.m"; "scripts/save_tracking_results.m"];
srcFiles = dir(fullfile(projectPaths.src,'**','*.m'));
for k = 1:numel(srcFiles)
    absoluteName = fullfile(srcFiles(k).folder,srcFiles(k).name);
    relativeName = erase(string(absoluteName),string(thisDir)+filesep);
    codeFiles(end+1,1) = replace(relativeName,filesep,"/");
end
codeFiles = sort(codeFiles);
codeHashes = strings(size(codeFiles));
for k = 1:numel(codeFiles)
    codeHashes(k) = study_hash(fullfile(thisDir,codeFiles(k)),"file");
end
workerCount = 0;
if USE_PARALLEL
    pool = gcp('nocreate');
    workerCount = pool.NumWorkers;
end
settings = struct('mission',missionCfg,'measurements',measCfg, ...
    'cost',costCfg,'costFlags',costFlags,'P0',P_0,'Q',Q_k,'R',R_k, ...
    'useScreening',useScreening, ...
    'visibilityDefinition',"center_referenced_max_v1", ...
    'sun_exclusion',sun_exclusion, ...
    'moon_exclusion',moon_exclusion,'earth_exclusion',earth_exclusion, ...
    'theta0',theta0,'i_sun',i_sun,'mu',mu,'LU',LU,'TU',TU, ...
    'EKF_DT',EKF_DT,'slotsPerOrbit',slots_per_orbit, ...
    'slotDefinition',"equal_time_no_endpoint_v1", ...
    'feBudgetDefinition',"admitted_completed_evaluations_v1", ...
    'parallelBoOverflowPolicy',"truncate_after_budget_v1", ...
    'odeRelTol',ode_opts.RelTol,'odeAbsTol',ode_opts.AbsTol, ...
    'useParallel',USE_PARALLEL,'workerCount',workerCount);

runState = struct();
runState.schemaVersion = 2;
runState.studyID = STUDY_ID;
runState.optimizer = OPTIMIZER_MODE;
runState.runTag = string(RUN_TAG);
runState.optimizerSeed = seedVal;
runState.measurementNoiseSeed = measCfg.noiseSeed;
runState.maxEvaluations = FE_BUDGET;
runState.searchEvaluationBudget = FE_BUDGET;
runState.settings = settings;
runState.truthInfo = truthInfo;
runState.catalogHash = catalogHash;
runState.truthHash = study_hash({t_target_ekf,s_target_ekf});
runState.codeHash = study_hash({codeFiles,codeHashes});
runState.matlabVersion = string(version);
runState.platform = string(computer);
runState.host = string(getenv('COMPUTERNAME'));
if runState.host == "", runState.host = string(getenv('HOSTNAME')); end
runState.toolboxes = ver;
runState.comparison = struct('settings',settings, ...
    'budget',FE_BUDGET,'catalogHash',catalogHash, ...
    'truthHash',runState.truthHash,'codeHash',runState.codeHash, ...
    'matlabVersion',runState.matlabVersion);
runState.comparisonKey = study_hash(runState.comparison);
runState.status = "running";
runState.termination = "running";
runState.validationStatus = "not_run";
runState.validationEvaluations = 0;
runState.created = string(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
runStateFile = fullfile(DataDir,'optimization_run.mat');
save(runStateFile,'runState','-v7');

RunTimer = tic;
solverError = [];
solverExitFlag = NaN;
solverOutput = struct();

% ---------------- Optimization ----------------
try
    switch upper(OPTIMIZER_MODE)

        case 'GA'
            safe_printf('Starting Genetic Algorithm...\n');

            nVars = nVars_common;
            LB = LB_common;
            UB = UB_common;
            IntCon = 1:nVars;
            pop = 60;

            assert(mod(FE_BUDGET,pop) == 0, ...
                'GA FE_BUDGET must be divisible by PopulationSize.');

            % Generation 0 evaluates the initial population.
            gaMaxGenerations = FE_BUDGET/pop - 1;

            options = optimoptions('ga', ...
                'UseParallel', USE_PARALLEL, ...
                'UseVectorized', false, ...
                'Display', 'off', ...
                'PopulationSize', pop, ...
                'EliteCount', 0, ...
                'MaxGenerations', gaMaxGenerations, ...
                'MaxStallGenerations', Inf, ...
                'FunctionTolerance', 0, ...
                'ConstraintTolerance', 0, ...
                'FitnessLimit', -Inf, ...
                'OutputFcn', @(options,state,flag) ...
                    ga_outfun(options,state,flag,FE_BUDGET));

            solverSettingsText = string(evalc('disp(options)'));
            [x_best, min_cost, solverExitFlag, solverOutput] = ga( ...
                ObjFcn, nVars, [], [], [], [], ...
                LB, UB, [], IntCon, options);

            solverFunccount = solverOutput.funccount;
            incumbent = getappdata(0,'OPT_GA_BEST');
            assert(~isempty(incumbent.x) && isfinite(incumbent.J), ...
                'GA did not record a finite feasible incumbent.');
            x_best = incumbent.x;
            min_cost = incumbent.J;
            history = get_fe_history();
            if isempty(history)
                actualEvals = solverFunccount;
            else
                % Keep the final checkpoint count and solver total separately.
                % A difference is not automatically a nonsearch evaluation.
                actualEvals = history.fe(end);
            end

        case 'PSO'
            safe_printf('Starting Particle Swarm Optimization...\n');

            nVars = nVars_common;
            LB = LB_common;
            UB = UB_common;
            swarm = 60;

            assert(mod(FE_BUDGET,swarm) == 0, ...
                'PSO FE_BUDGET must be divisible by SwarmSize.');

            % Iteration 0 evaluates the initial swarm.
            psoMaxIterations = FE_BUDGET/swarm - 1;

            options = optimoptions('particleswarm', ...
                'UseParallel', USE_PARALLEL, ...
                'UseVectorized', false, ...
                'Display', 'off', ...
                'SwarmSize', swarm, ...
                'MaxIterations', psoMaxIterations, ...
                'MaxStallIterations', psoMaxIterations + 1, ...
                'FunctionTolerance', 0, ...
                'OutputFcn', @(values,state) ...
                    pso_outfun(values,state,FE_BUDGET));

            solverSettingsText = string(evalc('disp(options)'));
            [x_best, min_cost, solverExitFlag, solverOutput] = ...
                particleswarm(ObjFcn, nVars, LB, UB, options);

            x_best = round(x_best);
            actualEvals = solverOutput.funccount;
            solverFunccount = actualEvals;
            history = get_fe_history();

        case 'BAYESIAN'
            safe_printf('Starting Bayesian Optimization...\n');

            vars = [];
            for i = 1:num_obs_cfg
                vars = [vars, ...
                    optimizableVariable(['Orbit',num2str(i)], ...
                        [1, num_orbits], 'Type','integer'), ...
                    optimizableVariable(['Slot',num2str(i)], ...
                        [1, slots_per_orbit], 'Type','integer')];
            end

            boSettings = struct('UseParallel',USE_PARALLEL, ...
                'IsObjectiveDeterministic',true, ...
                'AcquisitionFunctionName',"expected-improvement-plus", ...
                'MaxObjectiveEvaluations',FE_BUDGET, ...
                'BudgetPolicy',"parallel_overflow_truncate_v1", ...
                'OutputFcn',"bo_outfun",'PlotFcn',[]);
            solverSettingsText = string(evalc('disp(boSettings)'));
            results = bayesopt(ObjFcn, vars, ...
                'UseParallel', USE_PARALLEL, ...
                'IsObjectiveDeterministic', true, ...
                'AcquisitionFunctionName', 'expected-improvement-plus', ...
                'MaxObjectiveEvaluations', FE_BUDGET, ...
                'OutputFcn', @(boResults,state) ...
                    bo_outfun(boResults,state,FE_BUDGET,RunTimer), ...
                'PlotFcn', []);

            solverFunccount = results.NumObjectiveEvaluations;
            assert(solverFunccount >= FE_BUDGET, ...
                'Bayesian optimization stopped before the requested FE budget.');

            % Parallel bayesopt can finish evaluations that were already in
            % flight when the budget boundary was reached. Only the first
            % FE_BUDGET completed evaluations are admitted to the comparison.
            budgetObjective = results.ObjectiveTrace(1:FE_BUDGET);
            budgetX = results.XTrace(1:FE_BUDGET,:);
            budgetErrors = results.ErrorTrace(1:FE_BUDGET);
            objectiveErrorCount = nnz(budgetErrors == 1);

            objectiveForBest = budgetObjective;
            objectiveForBest(~isfinite(objectiveForBest)) = Inf;
            [min_cost,bestIdx] = min(objectiveForBest);
            assert(isfinite(min_cost), ...
                'No finite Bayesian objective value within the FE budget.');
            x_best = table2array(budgetX(bestIdx,:));

            actualEvals = FE_BUDGET;
            parallelOverflowEvals = solverFunccount - FE_BUDGET;
            history = table( ...
                (1:FE_BUDGET)', ...
                cummin(objectiveForBest(:)), ...
                'VariableNames', {'fe','bestJ'});

            budgetRuntime_s = getappdata(0,'OPT_BO_BUDGET_RUNTIME');
            if isempty(budgetRuntime_s) || ~isfinite(budgetRuntime_s)
                if numel(results.IterationTimeTrace) >= FE_BUDGET
                    budgetRuntime_s = sum(results.IterationTimeTrace(1:FE_BUDGET));
                else
                    budgetRuntime_s = NaN;
                end
            end

            safe_printf(['Bayesian FE budget admitted: %d | solver calls: %d ' ...
                '| parallel overflow: %d\n'], ...
                actualEvals,solverFunccount,parallelOverflowEvals);

        case 'ABC'
            safe_printf('Starting Artificial Bee Colony Optimization...\n');

            LB = LB_common;
            UB = UB_common;

            abc_opts.ColonySize      = 60;
            abc_opts.MaxEvals        = FE_BUDGET;
            abc_opts.Limit           = 20;
            abc_opts.StallIters      = inf;
            abc_opts.SlotsPerOrbit   = slots_per_orbit;
            abc_opts.UseParallel     = USE_PARALLEL;
            abc_opts.UseParallelInit = USE_PARALLEL;
            abc_opts.Logger          = @safe_printf;

            solverSettingsText = string(evalc('disp(abc_opts)'));
            [x_best, min_cost, actualEvals, history] = ...
                abc_discrete(ObjFcn, LB, UB, abc_opts);
            solverFunccount = actualEvals;

        case 'ACO'
            safe_printf('Starting Ant Colony Optimization...\n');

            LB = LB_common;
            UB = UB_common;

            aco_opts.nAnts              = 60;
            aco_opts.MaxEvals           = FE_BUDGET;
            aco_opts.alpha              = 1.0;
            aco_opts.beta               = 1.0;
            aco_opts.rho                = 0.2;
            aco_opts.Q                  = 1.0;
            aco_opts.UseParallel        = USE_PARALLEL;
            aco_opts.TauMin             = 1e-12;
            aco_opts.UseIterBestDeposit = true;
            aco_opts.IterBestWeight     = 1.0;
            aco_opts.StallIters         = inf;
            aco_opts.Logger             = @safe_printf;

            solverSettingsText = string(evalc('disp(aco_opts)'));
            [x_best, min_cost, actualEvals, history] = ...
                aco_discrete(ObjFcn, LB, UB, aco_opts);
            solverFunccount = actualEvals;

        otherwise
            error("Unknown OPTIMIZER_MODE: %s", OPTIMIZER_MODE);
    end
catch ME
    solverError = ME;
end

% ---------------- Runtime / final optimization results ----------------
SolverWallRuntime_s = toc(RunTimer);
if OPTIMIZER_MODE ~= "BAYESIAN" || ~isfinite(budgetRuntime_s)
    budgetRuntime_s = SolverWallRuntime_s;
end
safe_printf('Budget Runtime: %.2f seconds\n', budgetRuntime_s);
if OPTIMIZER_MODE == "BAYESIAN" && solverFunccount > actualEvals
    safe_printf('Solver Wall Runtime (including BO overflow): %.2f seconds\n', ...
        SolverWallRuntime_s);
end

% Preserve a partial callback history when GA/PSO throws.
if isempty(history) && ismember(OPTIMIZER_MODE,["GA","PSO"])
    history = get_fe_history();
    if ~isempty(history), actualEvals = history.fe(end); end
end
runState.nEvaluations = actualEvals;
runState.searchFunctionEvaluations = actualEvals;
runState.solverFunctionEvaluations = solverFunccount;
runState.solverCallDifference = solverFunccount-actualEvals;
runState.postSearchFunctionEvaluations = ...
    max(0,solverFunccount-actualEvals);
runState.parallelOverflowEvaluations = parallelOverflowEvals;
runState.bestX = x_best;
runState.bestJ = min_cost;
runState.history = history;
runState.runtime_s = budgetRuntime_s;
runState.budgetRuntime_s = budgetRuntime_s;
runState.solverWallRuntime_s = SolverWallRuntime_s;
runState.solverExitFlag = solverExitFlag;
runState.solverOutput = solverOutput;
runState.solverSettingsText = solverSettingsText;
runState.objectiveErrorCount = objectiveErrorCount;

if ~isempty(solverError)
    runState.status = "solver_failed";
    runState.termination = "solver_failed";
    runState.error = string(getReport(solverError,'extended','hyperlinks','off'));
    save(runStateFile,'runState','-v7');
    diary off
    rethrow(solverError);
end
if actualEvals ~= FE_BUDGET
    runState.status = "solver_failed";
    runState.termination = "budget_not_reached";
    save(runStateFile,'runState','-v7');
    diary off
    error('Optimizer stopped at %g FE; expected exactly %d FE.', ...
        actualEvals,FE_BUDGET);
end
runState.termination = "budget_reached";
runState.status = "optimized";
save(runStateFile,'runState','-v7');
try
    assert(objectiveErrorCount == 0, ...
        'Objective errors occurred; inspect the saved run before comparison.');
    assert(~isempty(history) && all(isfinite(history.fe)) && ...
        all(history.fe > 0 & history.fe == round(history.fe)) && ...
        all(diff(history.fe) > 0) && all(isfinite(history.bestJ)) && ...
        all(diff(history.bestJ) <= 1e-12*max(1,abs(history.bestJ(1:end-1)))) && ...
        history.fe(end) == actualEvals && ...
        abs(history.bestJ(end)-min_cost) <= 1e-9*max(1,abs(min_cost)), ...
        'Invalid convergence history or inconsistent best solution.');

    x_best = round(x_best);
    orbit_indices = x_best(1:2:end);
    slot_indices = x_best(2:2:end);
    num_obs = numel(orbit_indices);
    observer_ICs = zeros(num_obs,6);
    for k = 1:num_obs
        observer_ICs(k,:) = orbit_database{orbit_indices(k)}(slot_indices(k),:);
    end
    observers = table((1:num_obs)',orbit_indices(:),slot_indices(:), ...
        string(T1.orbitFamily(orbit_indices)), ...
        tf(orbit_indices),stabilities(orbit_indices),observer_ICs, ...
        'VariableNames',{'observer_id','orbit_index','slot_index', ...
        'orbit_family','period_TU','stability_index','initial_state'});
    if ismember('sourceFile',T1.Properties.VariableNames)
        observers.source_file = string(T1.sourceFile(orbit_indices));
    end
    if ismember('orbitID',T1.Properties.VariableNames)
        observers.orbit_id = string(T1.orbitID(orbit_indices));
    end

    % One diagnostic EKF pass, outside the search budget.
    runState.validationStatus = "running";
    runState.validationEvaluations = 1;
    save(runStateFile,'runState','-v7');
    validationTimer = tic;
    [s_ekf,cov,screeningCount_final,availableObsCount] = cr3bp_ekf( ...
        observer_ICs,s_target_ekf,t_target_ekf,P_0,Q_k,R_k,mu,LU, ...
        sunFcn,sun_exclusion,moon_exclusion,earth_exclusion,useScreening,measCfg);
    runState.validationRuntime_s = toc(validationTimer);
    runState.status = "completed";
    runState = save_tracking_results(DataDir,runState, ...
        t_target_ekf,s_target_ekf,s_ekf,cov,availableObsCount, ...
        screeningCount_final,observers);
catch ME
    runState.status = "validation_failed";
    runState.validationStatus = "failed";
    runState.error = string(getReport(ME,'extended','hyperlinks','off'));
    save(runStateFile,'runState','-v7');
    diary off
    rethrow(ME);
end

safe_printf(['Search FE = %d/%d | solver calls = %d ' ...
    '(extra solver calls = %d, BO overflow = %d) | bestJ = %.12g | %s\n'], ...
    actualEvals,FE_BUDGET,solverFunccount, ...
    runState.postSearchFunctionEvaluations, ...
    runState.parallelOverflowEvaluations,min_cost,runState.termination);
safe_printf('Saved data only: %s\n',DataDir);
safe_printf('RUN END: %s\n',string(datetime('now')));
diary off
if MAKE_PLOTS
    try
        preview_study_run(DataDir); % Display only, after data have been saved.
    catch ME
        warning('Study:PreviewFailed','Preview failed: %s',ME.message);
    end
end
return;

% ---------------- Helper functions ----------------

function cacheKey = make_transfer_cache_key(missionCfg)
    tr = missionCfg.transfer;
    lt = tr.lowthrust;
    endpointHash = string(study_hash({tr.fixedDepartureState,tr.fixedTargetState}));
    shortHash = extractBefore(endpointHash, min(9,strlength(endpointHash)+1));

    cacheKey = sprintf('lt_fixed_%s_dt%s_tf%s', ...
        char(shortHash), ...
        local_num_str(get_field_or_default(tr, 'dt', 0)), ...
        local_num_str(get_field_or_default(lt, 'tf_guess', 0)));

    cacheKey = regexprep(cacheKey, '[^A-Za-z0-9_]', '_');
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


function safe_printf(varargin)
    try
        if ~isempty(varargin) && isnumeric(varargin{1}) && isscalar(varargin{1})
            fid = varargin{1};
            msg = sprintf(varargin{2:end});
        else
            fid = 1;
            msg = sprintf(varargin{:});
        end

        try
            fprintf(fid, '%s', msg);
            if isempty(msg) || msg(end) ~= newline
                fprintf(fid, '\n');
            end
        catch
        end

        append_fallback_output(msg);
    catch
    end
end


function append_fallback_output(msg)
    try
        fallbackFile = getenv('SAFE_FALLBACK_FILE');
        if isempty(fallbackFile)
            return;
        end

        fid = fopen(fallbackFile, 'a');
        if fid < 0
            return;
        end

        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, '%s', msg);
        if isempty(msg) || msg(end) ~= newline
            fprintf(fid, '\n');
        end
    catch
    end
end    

function [state,options,optchanged] = ga_outfun(options,state,flag,FE_BUDGET)
    optchanged = false;
    if strcmp(flag,'init') || strcmp(flag,'iter')
        incumbent = getappdata(0,'OPT_GA_BEST');
        % Fitness contains objective values and is Inf for infeasible
        % individuals in this integer GA. Score can contain penalties.
        values = state.Score(:);
        if isfield(state,'Fitness'), values = state.Fitness(:); end
        values(~isfinite(values)) = Inf;
        [generationBest,k] = min(values);
        if ~isempty(generationBest) && generationBest < incumbent.J
            incumbent.J = generationBest;
            incumbent.x = state.Population(k,:);
            setappdata(0,'OPT_GA_BEST',incumbent);
        end
        append_fe_history(state.FunEval,incumbent.J);
        safe_printf('GA gen %3d | FE = %5d | bestJ = %.12g\n', ...
            state.Generation,state.FunEval,incumbent.J);
    end
    if state.FunEval >= FE_BUDGET
        state.StopFlag = 'Function evaluation budget reached';
    end
end

function stop = pso_outfun(values,state,FE_BUDGET)
    stop = false;

    if strcmp(state,'init') || strcmp(state,'iter')
        append_fe_history(values.funccount, values.bestfval);

        safe_printf( ...
            'PSO iter %3d | FE = %5d | bestJ = %.12g\n', ...
            values.iteration, values.funccount, values.bestfval);
    end

    stop = values.funccount >= FE_BUDGET;
end

function stop = bo_outfun(results,state,FE_BUDGET,RunTimer)
    stop = false;

    if strcmp(state,'iteration') && ...
            results.NumObjectiveEvaluations >= FE_BUDGET
        tBudget = getappdata(0,'OPT_BO_BUDGET_RUNTIME');
        if isempty(tBudget) || ~isfinite(tBudget)
            setappdata(0,'OPT_BO_BUDGET_RUNTIME',toc(RunTimer));
        end
        % Prevent new BO work from being scheduled. Evaluations that are
        % already active can still finish and are recorded as overflow.
        stop = true;
    end
end

function reset_fe_history()
    setappdata(0,'OPT_FE_HISTORY',zeros(0,2));
    setappdata(0,'OPT_GA_BEST',struct('J',Inf,'x',[]));
    setappdata(0,'OPT_BO_BUDGET_RUNTIME',NaN);
end

function append_fe_history(fe, bestJ)
    H = getappdata(0, 'OPT_FE_HISTORY');
    if isempty(H)
        H = zeros(0,2);
    end

    if isempty(H) || fe > H(end,1)
        H(end+1,:) = [fe, bestJ];
    elseif fe == H(end,1)
        H(end,2) = min(H(end,2), bestJ);
    end

    setappdata(0, 'OPT_FE_HISTORY', H);
end

function T = get_fe_history()
    H = getappdata(0, 'OPT_FE_HISTORY');
    if isempty(H)
        T = table([], [], 'VariableNames', {'fe','bestJ'});
    else
        T = array2table(H, 'VariableNames', {'fe','bestJ'});
    end
end