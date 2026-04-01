% ---- run_opt.m ---- %
clear; close all; clc;

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

% Ensure all project functions are on path (for local + batch runs)
addpath(genpath(thisDir));

catalogPath = fullfile(thisDir, 'JPL_CR3BP_OrbitCatalog.mat');
S = load(catalogPath);
CatalogDir = thisDir;
T1 = S.T;
t_lg = S.t_lg; 
s_lg = S.s_lg; 

% ---------------- Optimizer inputs ----------------
% Options: 'GA', 'PSO', 'BAYESIAN', 'GAMULTIOBJ', 'DMOPSO', 'ABC', 'ACO'
OPTIMIZER_MODE = 'GA';

envMode = getenv("OPTIMIZER_MODE");
if ~isempty(envMode)
    OPTIMIZER_MODE = envMode;
end
OPTIMIZER_MODE = upper(string(OPTIMIZER_MODE));

% Stopping Criteria (max iterations for all except Bayesian)
MAX_ITERS = 10;
v = getenv("MAX_ITERS"); if ~isempty(v), MAX_ITERS = str2double(v); end

MAX_EVALS = 100;
v = getenv("MAX_EVALS"); if ~isempty(v), MAX_EVALS = str2double(v); end

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

OrbitCacheDir = fullfile(CatalogDir, "orbit_cache");
if ~exist(OrbitCacheDir,'dir'), mkdir(OrbitCacheDir); end

catalogFile = "JPL_CR3BP_OrbitCatalog.mat";
orbitDbCacheFile = fullfile(OrbitCacheDir, ...
    sprintf('orbit_database_slots_%d.mat', slots_per_orbit));

rebuildOrbitDb = true;

if isfile(orbitDbCacheFile)
    try
        C = load(orbitDbCacheFile, 'orbit_database', 'cacheMeta');
        if isfield(C, 'orbit_database') && numel(C.orbit_database) == num_orbits
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

        t_slots = linspace(0, period, slots_per_orbit)';

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
sun_min_deg  = 20;
moon_min_deg = 10;

sun_min  = deg2rad(sun_min_deg);
moon_min = deg2rad(moon_min_deg);

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

% set up data logging (in-memory only)
dq = parallel.pool.DataQueue;
assignin('base', 'OptimizationLog', {});
afterEach(dq, @(data) append_log(data));

function append_log(data)
    logCell = evalin('base', 'OptimizationLog');
    logCell{end+1,1} = data;
    assignin('base', 'OptimizationLog', logCell);
end

opt_flag          = 'SOO';
const_stabilities = parallel.pool.Constant(stabilities); 
const_orbit_db    = parallel.pool.Constant(orbit_database); 

% ---------------- Mission type ----------------
MISSION_TYPE = "LOW_THRUST_TRANSFER";

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
        missionCfg.gateway.s0 = [1.02202108343387, 0, -0.182096487798513, ...
                                 0, -0.103255420206012, 0]';
        missionCfg.gateway.period   = 1.51110546287394;
        missionCfg.gateway.dt       = 0.001;
        missionCfg.gateway.Nperiods = 1;

    case "PERIODIC_ORBIT"
        missionCfg.periodic.orbitIndex = 1;
        missionCfg.periodic.dt         = 0.001;
        missionCfg.periodic.Nperiods   = 1;

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

    otherwise
        error("Unknown MISSION_TYPE: %s", missionCfg.type);
end

% --- override number of periods from environment ---
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
vseed = getenv("SEED"); if isempty(vseed), vseed = "0"; end
seedVal = str2double(vseed); if isnan(seedVal), seedVal = 0; end
rng(seedVal, 'twister');

RUN_TAG = sprintf('%s_scr%d_J%d%d%d_seed%03d', char(OPTIMIZER_MODE), ...
    double(useScreening), double(costFlags.J1), double(costFlags.J2), double(costFlags.J3), seedVal);

if strlength(string(RunDir)) == 0
    ts = string(datetime('now','Format','yyyyMMdd_HHmmss'));

    switch upper(string(missionCfg.type))
        case "LOW_THRUST_TRANSFER"
            missionCode = "lt";
        case "LUNAR_GATEWAY"
            missionCode = "lg";
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

    RunsRoot   = fullfile(thisDir, "runs");
    BatchDir   = fullfile(RunsRoot, ts);
    MissionDir = fullfile(BatchDir, missionCode);
    RunDir     = fullfile(MissionDir, caseName);

    if ~exist(RunsRoot,'dir'), mkdir(RunsRoot); end
    if ~exist(BatchDir,'dir'), mkdir(BatchDir); end
    if ~exist(MissionDir,'dir'), mkdir(MissionDir); end
    if ~exist(RunDir,'dir'), mkdir(RunDir); end

    cd(RunDir);
end

FigDir  = fullfile(RunDir, "figs");
DataDir = fullfile(RunDir, "data");
LogDir  = fullfile(RunDir, "logs");

TransferCacheDir = fullfile(CatalogDir, "transfer_cache");

if ~exist(FigDir,'dir'), mkdir(FigDir); end
if ~exist(DataDir,'dir'), mkdir(DataDir); end
if ~exist(LogDir,'dir'), mkdir(LogDir); end
if ~exist(TransferCacheDir,'dir'), mkdir(TransferCacheDir); end

switch upper(string(missionCfg.type))
    case "LOW_THRUST_TRANSFER"
        missionCodeShort = "lt";
    case "LUNAR_GATEWAY"
        missionCodeShort = "lg";
    case "PERIODIC_ORBIT"
        missionCodeShort = "po";
    otherwise
        missionCodeShort = lower(string(missionCfg.type));
end

FILE_TAG = sprintf('%s_%s_o%d', char(RUN_TAG), char(missionCodeShort), missionCfg.optimization.numObservers);

% ---------------- Output / log files ----------------
EXCEL_FILE = fullfile(DataDir, sprintf('ExperimentSummary_%s.xlsx', FILE_TAG));
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
    r_ang = 1e-8;
    Q_k = diag([q_pos q_pos q_pos q_vel q_vel q_vel]);
    R_k = diag([r_ang r_ang]);

elseif strcmp(MISSION_TYPE, "LUNAR_GATEWAY") || strcmp(MISSION_TYPE, "PERIODIC_ORBIT")
    pos_var  = (1 / LU)^2;
    vel_var  = (10 / (VU * 1000))^2;
    P_0 = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);

    q_pos = 1e-8;
    q_vel = 1e-8;
    r_ang = 1e-8;
    Q_k = diag([q_pos q_pos q_pos q_vel q_vel q_vel]);
    R_k = diag([r_ang r_ang]);
end

% ---------------- Cost thresholds ----------------
costCfg = struct();
costCfg.weights = [1, 1, 0.1];

switch upper(string(missionCfg.type))
    case "LOW_THRUST_TRANSFER"
        costCfg.pos_rmse_acc = 10 / LU;
        costCfg.vel_rmse_acc = 0.01 / VU;
        costCfg.sigma_pos_acc = 10 / LU;
        costCfg.sigma_vel_acc = 0.01 / VU;
        costCfg.stability_acc = 1.0;

    case "LUNAR_GATEWAY"
        costCfg.pos_rmse_acc = 0.1 / LU;
        costCfg.vel_rmse_acc = 1.0e-4 / VU;
        costCfg.sigma_pos_acc = 0.1 / LU;
        costCfg.sigma_vel_acc = 1.0e-4 / VU;
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

if contains(string(missionCfg.type), "TRANSFER") && useTransferCache
    cacheKey  = make_transfer_cache_key(missionCfg, slots_per_orbit);
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
                safe_printf('Deleted corrupt transfer cache:\n  %s\n', cacheFile);
            catch MEdel
                safe_printf(2, 'WARNING: failed to delete corrupt transfer cache: %s\n', MEdel.message);
            end
        end
    end

    if ~loadedFromCache
        safe_printf('No valid cached transfer found. Computing transfer truth.\n');

        [t_target, s_target, truthInfo] = build_target_truth( ...
            missionCfg, T1, orbit_database, times, states, mu, ode_opts);

        cacheMeta = struct();
        cacheMeta.cacheKey        = cacheKey;
        cacheMeta.missionType     = string(missionCfg.type);
        cacheMeta.created         = string(datetime('now'));
        cacheMeta.slots_per_orbit = slots_per_orbit;
        cacheMeta.mu              = mu;

        tmpFile = fullfile(TransferCacheDir, cacheKey + "_t.mat");

        try
            if isfile(tmpFile)
                delete(tmpFile);
            end

            if isfile(cacheFile)
                delete(cacheFile);
            end

            save(tmpFile, 't_target', 's_target', 'truthInfo', 'cacheMeta', '-v7.3');
            movefile(tmpFile, cacheFile, 'f');

            safe_printf('Saved transfer truth cache to:\n  %s\n', cacheFile);

        catch ME
            safe_printf(2, 'WARNING: failed to save transfer cache: %s\n', ME.message);

            try
                if isfile(tmpFile)
                    delete(tmpFile);
                end
            catch
            end
        end
    end
else
    [t_target, s_target, truthInfo] = build_target_truth( ...
        missionCfg, T1, orbit_database, times, states, mu, ode_opts);
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

% Objective Function Wrapper
ObjFcn = @(x) objective_wrapper(x, orbit_database, stabilities, s_target_ekf, t_target_ekf, P_0, Q_k, R_k, mu, LU, ...
    sunFcn, sun_min, moon_min, opt_flag, OPTIMIZER_MODE, dq, useScreening, costFlags, costCfg);

RunTimer = tic;

% ---------------- Optimization ----------------
switch upper(OPTIMIZER_MODE)

    case 'GA'
        safe_printf('Starting Genetic Algorithm...\n');
        nVars = nVars_common;
        LB = LB_common;
        UB = UB_common;
        IntCon = 1:nVars;

        pop = 60;

        options = optimoptions('ga', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'PopulationSize', pop, ...
            'MaxGenerations', MAX_ITERS, ...
            'MaxStallGenerations', Inf, ...
            'FunctionTolerance', 0, ...
            'ConstraintTolerance', 0, ...
            'FitnessLimit', -Inf, ...
            'OutputFcn', @ga_outfun);

        [x_best, min_cost, ~, ~, population, scores] = ga(ObjFcn, nVars, [], [], [], [], LB, UB, [], IntCon, options);
        J_check = ObjFcn(x_best);

        safe_printf('ga returned min_cost = %.12f\n', min_cost);
        safe_printf('reevaluated J(x_best) = %.12f\n', J_check);

        [bestFinalScore, idxBestFinal] = min(scores);
        safe_printf('best score in final population = %.12f\n', bestFinalScore);
        safe_disp('x_best returned by ga:');
        safe_disp(x_best);
        safe_disp('best individual in final population:');
        safe_disp(population(idxBestFinal,:));

    case 'PSO'
        safe_printf('Starting Particle Swarm Optimization...\n');
        nVars = nVars_common;
        LB = LB_common;
        UB = UB_common;

        swarm = 60;

        options = optimoptions('particleswarm', ...
            'UseParallel', true, ...
            'Display', 'off', ...
            'SwarmSize', swarm, ...
            'MaxIterations', MAX_ITERS, ...
            'OutputFcn', @pso_outfun);

        [x_best, min_cost] = particleswarm(ObjFcn, nVars, LB, UB, options);
        x_best = round(x_best);

    case 'BAYESIAN'
        safe_printf('Starting Bayesian Optimization...\n');

        vars = [];
        for i = 1:num_obs_cfg
            vars = [vars, ...
                optimizableVariable(['Orbit',num2str(i)], [1, num_orbits], 'Type','integer'), ...
                optimizableVariable(['Slot', num2str(i)], [1, slots_per_orbit], 'Type','integer')]; 
        end

        if isappdata(0, 'BAYES_EVAL_COUNTER'), rmappdata(0, 'BAYES_EVAL_COUNTER'); end
        if isappdata(0, 'BAYES_BEST_COST'),    rmappdata(0, 'BAYES_BEST_COST');    end
        setappdata(0, 'BAYES_EVAL_COUNTER', 0);
        setappdata(0, 'BAYES_BEST_COST', inf);
        setappdata(0, 'BAYES_OBJFCN', ObjFcn);

        results = bayesopt(@bayes_objective_with_logging, vars, ...
            'UseParallel', false, ...
            'IsObjectiveDeterministic', false, ...
            'MaxObjectiveEvaluations', MAX_EVALS);

        x_best   = table2array(results.XAtMinObjective);
        min_cost = results.MinObjective;

        if isappdata(0, 'BAYES_OBJFCN'),       rmappdata(0, 'BAYES_OBJFCN');       end
        if isappdata(0, 'BAYES_EVAL_COUNTER'), rmappdata(0, 'BAYES_EVAL_COUNTER'); end
        if isappdata(0, 'BAYES_BEST_COST'),    rmappdata(0, 'BAYES_BEST_COST');    end

    case 'GAMULTIOBJ'
        safe_printf('Starting Multi-Objective Genetic Algorithm (NSGA-II)...\n');

        nVars = nVars_common;
        LB = double(LB_common);
        UB = double(UB_common);
        IntCon = 1:nVars;

        pop = 60;

        options = optimoptions('gamultiobj', ...
            'PopulationSize', pop, ...
            'MaxGenerations', MAX_ITERS, ...
            'ParetoFraction', 0.5, ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'PlotFcn', @gaplotpareto);

        [x_best, fval] = gamultiobj(ObjFcn, nVars, [], [], [], [], LB, UB, [], IntCon, options);

    case 'DMOPSO'
        safe_printf('Starting Custom Multi-Objective PSO...\n');

        nVars = nVars_common;
        LB = double(LB_common);
        UB = double(UB_common);

        swarmSize  = 60;
        maxIter    = MAX_ITERS;
        stallIters = inf;

        [archive_X, archive_F] = dmopso(ObjFcn, nVars, LB, UB, swarmSize, maxIter, stallIters);
        fval   = archive_F;
        x_best = archive_X;

    case 'ABC'
        safe_printf('Starting Artificial Bee Colony Optimization...\n');

        LB = LB_common;
        UB = UB_common;

        abc_opts.ColonySize      = 60;
        abc_opts.MaxIters        = MAX_ITERS;
        abc_opts.Limit           = 20;
        abc_opts.StallIters      = inf;
        abc_opts.SlotsPerOrbit   = slots_per_orbit;
        abc_opts.UseParallel     = true;
        abc_opts.UseParallelInit = true;
        abc_opts.Logger          = @safe_printf;

        [x_best, min_cost] = abc_discrete(ObjFcn, LB, UB, abc_opts);

    case 'ACO'
        safe_printf('Starting Ant Colony Optimization...\n');

        LB = LB_common;
        UB = UB_common;

        aco_opts.nAnts                = 60;
        aco_opts.MaxIters             = MAX_ITERS;
        aco_opts.alpha                = 1.0;
        aco_opts.beta                 = 1.0;
        aco_opts.rho                  = 0.2;
        aco_opts.Q                    = 1.0;
        aco_opts.UseParallel          = true;
        aco_opts.TauMin               = 1e-12;
        aco_opts.UseIterBestDeposit   = true;
        aco_opts.IterBestWeight       = 1.0;
        aco_opts.StallIters           = inf;
        aco_opts.Logger               = @safe_printf;

        [x_best, min_cost] = aco_discrete(ObjFcn, LB, UB, aco_opts);

    otherwise
        error("Unknown OPTIMIZER_MODE: %s", OPTIMIZER_MODE);
end

% ---------------- Runtime / final optimization results ----------------
TotalRuntime = toc(RunTimer);
safe_printf('Total Runtime: %.2f seconds\n', TotalRuntime);

if strcmpi(opt_flag, 'SOO')
    safe_printf('\n--- FINAL RESULTS (%s) ---\n', OPTIMIZER_MODE);
    safe_printf('Orbits: %s\n', mat2str(x_best(1:2:end)));
    safe_printf('Slots:  %s\n', mat2str(x_best(2:2:end)));
    safe_printf('Cost:   %.4f\n', min_cost);
    x_plot = x_best;
else
    f_min  = min(fval);
    f_max  = max(fval);
    f_norm = (fval - f_min) ./ (f_max - f_min);

    dist_to_utopia = sqrt(sum(f_norm.^2, 2));
    [~, idx_knee]  = min(dist_to_utopia);

    knee_costs = fval(idx_knee, :);
    knee_vars  = x_best(idx_knee, :);

    safe_printf('\n--- KNEE POINT (Balanced Solution) ---\n');
    safe_printf('Selected Row: %d\n', idx_knee);
    safe_printf('RMSE (Log):   %.4f\n', knee_costs(1));
    safe_printf('Det (Log):    %.4f\n', knee_costs(2));
    safe_printf('Stability:    %.4f\n', knee_costs(3));
    safe_printf('Orbits:       %s\n', mat2str(knee_vars(1:2:end)));
    safe_printf('Slots:        %s\n', mat2str(knee_vars(2:2:end)));
    x_plot = knee_vars;
end

safe_printf('RUN END: %s\n', string(datetime('now')));
drawnow;

% ---------------- Parallel pool cleanup ----------------
try
    p = gcp('nocreate');
    if ~isempty(p)
        delete(p);
    end
catch
end

drawnow;
pause(0.2);

% ---------------- Recompile results to plot ----------------
x_plot = round(x_plot);

orbit_indices = x_plot(1:2:end);
slot_indices  = x_plot(2:2:end);
num_obs = numel(orbit_indices);

for k = 1:num_obs
    orbit_indices(k) = max(1, min(orbit_indices(k), numel(orbit_database)));
    slot_indices(k)  = max(1, min(slot_indices(k), size(orbit_database{orbit_indices(k)},1)));
end

observer_ICs = zeros(num_obs,6);
for k = 1:num_obs
    observer_ICs(k,:) = orbit_database{orbit_indices(k)}(slot_indices(k),:);
end

availableObsCount = [];
try
    [s_ekf, cov, screeningCount_final, availableObsCount] = cr3bp_ekf(observer_ICs, s_target_ekf, t_target_ekf, ...
        P_0, Q_k, R_k, mu, LU, sunFcn, sun_min, moon_min, useScreening);
catch
    [s_ekf, cov, screeningCount_final] = cr3bp_ekf(observer_ICs, s_target_ekf, t_target_ekf, ...
        P_0, Q_k, R_k, mu, LU, sunFcn, sun_min, moon_min, useScreening);
end

safe_printf('\nFinal EKF screeningCount = %d\n', screeningCount_final);

availableObsCount = sanitize_obs_count_vector(availableObsCount, numel(t_target_ekf), num_obs);

% ---------------- Dense EKF replot grid ----------------
t_plot = t_truth(:);

[t_unique_ekf, idx_u_ekf] = unique(t_target_ekf(:));
s_ekf_unique = s_ekf(idx_u_ekf,:);

F_ekf_plot = griddedInterpolant(t_unique_ekf, s_ekf_unique, 'spline');
s_ekf_plot = F_ekf_plot(t_plot);

% ---------------- Observer metadata ----------------
familyColName = "orbitFamily";
obs_family = strings(num_obs,1);
if strlength(familyColName) > 0
    try
        famCol = T1.(familyColName);
        obs_family = string(famCol(orbit_indices));
    catch
        obs_family = strings(num_obs,1);
    end
end

obsTbl = table( ...
    (1:num_obs)', orbit_indices(:), slot_indices(:), obs_family(:), ...
    tf(orbit_indices), stabilities(orbit_indices), ...
    'VariableNames', {'observer_id','orbit_index','slot_index','orbit_family','period_TU','stability_index'} );

% ---------------- Trajectory plot ----------------
figW = 8;
figH = 6;
fig = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
             'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

ax = axes(fig);
hold(ax,'on');
box(ax,'on');
set(ax, 'TickLabelInterpreter','tex', 'Layer','top');
ax.Projection = 'orthographic';
view(ax, 32, 24);

cEKF = 'red';

hEKF = plot3(ax, s_ekf_plot(:,1), s_ekf_plot(:,2), s_ekf_plot(:,3), '-', ...
    'LineWidth', 2.4, 'Color', cEKF);

cmap = lines(max(1,num_obs));
hObs = gobjects(num_obs,1);
for k = 1:num_obs
    iOrb   = orbit_indices(k);
    s_raw  = states{iOrb};
    hObs(k) = plot3(ax, s_raw(:,1), s_raw(:,2), s_raw(:,3), '-', ...
        'Color', cmap(k,:), 'LineWidth', 1.8);
end

hDepOrb = gobjects(0);
hArrOrb = gobjects(0);
hStart  = gobjects(0);
hEnd    = gobjects(0);

isTransferMission = contains(string(missionCfg.type), "TRANSFER");

if isTransferMission
    depIdx = missionCfg.transfer.depOrbitIndex;
    arrIdx = missionCfg.transfer.arrOrbitIndex;

    s_dep_orb = states{depIdx};
    s_arr_orb = states{arrIdx};

    depBase = [0.00 0.45 0.74];
    arrBase = [0.85 0.33 0.10];
    depLight = 0.45*depBase + 0.55*[1 1 1];
    arrLight = 0.45*arrBase + 0.55*[1 1 1];

    hDepOrb = plot3(ax, s_dep_orb(:,1), s_dep_orb(:,2), s_dep_orb(:,3), '-', ...
        'Color', depLight, 'LineWidth', 1.5);

    hArrOrb = plot3(ax, s_arr_orb(:,1), s_arr_orb(:,2), s_arr_orb(:,3), '-', ...
        'Color', arrLight, 'LineWidth', 1.5);

    hStart = plot3(ax, s_truth(1,1), s_truth(1,2), s_truth(1,3), 'o', ...
        'MarkerSize', 9, 'MarkerFaceColor', depBase, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.0);

    hEnd = plot3(ax, s_truth(end,1), s_truth(end,2), s_truth(end,3), 's', ...
        'MarkerSize', 9, 'MarkerFaceColor', arrBase, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.0);

    if isfield(missionCfg.transfer, 'depSlot') && ~isempty(missionCfg.transfer.depSlot)
        depSlot = missionCfg.transfer.depSlot;
        depSlot = max(1, min(depSlot, size(orbit_database{depIdx},1)));
        depState0 = orbit_database{depIdx}(depSlot,:);

        plot3(ax, depState0(1), depState0(2), depState0(3), '^', ...
            'MarkerSize', 9, 'MarkerFaceColor', depBase, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.0);
    end

    if isfield(missionCfg.transfer, 'arrSlot') && ~isempty(missionCfg.transfer.arrSlot)
        arrSlot = missionCfg.transfer.arrSlot;
        arrSlot = max(1, min(arrSlot, size(orbit_database{arrIdx},1)));
        arrState0 = orbit_database{arrIdx}(arrSlot,:);

        plot3(ax, arrState0(1), arrState0(2), arrState0(3), 's', ...
            'MarkerSize', 9, 'MarkerFaceColor', arrBase, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.0);
    end
end

rM = [1-mu, 0, 0];
hM = plot3(ax, rM(1), rM(2), rM(3), 'ko', ...
    'MarkerSize',8, 'MarkerFaceColor',[0.70 0.70 0.70], 'LineWidth',1.0);

[xL1, xL2] = cr3bp_L1L2(mu);
hL1 = plot3(ax, xL1, 0, 0, 'k^', ...
    'MarkerSize',8, 'MarkerFaceColor',[0.85 0.85 0.85], 'LineWidth',1.0);
hL2 = plot3(ax, xL2, 0, 0, 'kv', ...
    'MarkerSize',8, 'MarkerFaceColor',[0.85 0.85 0.85], 'LineWidth',1.0);

for k = 1:num_obs
    iOrb   = orbit_indices(k);
    t_raw  = times{iOrb}(:);
    s_raw  = states{iOrb};
    Tper   = tf(iOrb);
    t_phase = (slot_indices(k)-1) / (slots_per_orbit-1) * Tper;
    [~, j] = min(abs(t_raw - t_phase));

    plot3(ax, s_raw(j,1), s_raw(j,2), s_raw(j,3), 'o', ...
        'MarkerSize',8, 'MarkerFaceColor', cmap(k,:), 'MarkerEdgeColor','k');
end

plot3(ax, s_truth(1,1), s_truth(1,2), s_truth(1,3), 'o', ...
    'MarkerSize',8, 'MarkerFaceColor', 'red', 'MarkerEdgeColor','k');

xlabel(ax,'x (LU)');
ylabel(ax,'y (LU)');
zlabel(ax,'z (LU)');

xl = xlim(ax);
yl = ylim(ax);
zl = zlim(ax);

xr = xl(2)-xl(1);
yr = yl(2)-yl(1);
zr = zl(2)-zl(1);

tickStep = max([xr, yr, zr]) / 5;
tickStep = max(tickStep, eps);

pow10 = 10^floor(log10(tickStep));
nice = tickStep / pow10;
if     nice <= 1
    nice = 1;
elseif nice <= 2
    nice = 2;
elseif nice <= 2.5
    nice = 2.5;
elseif nice <= 5
    nice = 5;
else
    nice = 10;
end
tickStep = nice * pow10;

xTick0 = ceil(xl(1)/tickStep)*tickStep;
yTick0 = ceil(yl(1)/tickStep)*tickStep;
zTick0 = ceil(zl(1)/tickStep)*tickStep;

ax.XTick = xTick0:tickStep:xl(2);
ax.YTick = yTick0:tickStep:yl(2);
ax.ZTick = zTick0:tickStep:zl(2);

legHandles = [hEKF; hObs(:)];
legLabels = cell(num_obs + 1, 1);
legLabels{1} = 'EKF estimate';
for k = 1:num_obs
    legLabels{1+k} = sprintf('Observer %d orbit', k);
end

if isTransferMission
    legHandles = [legHandles; hDepOrb; hArrOrb; hStart; hEnd];
    legLabels  = [legLabels; {'Departure orbit'; 'Arrival orbit'; 'Transfer start'; 'Transfer end'}];
end

legHandles = [legHandles; hM; hL1; hL2];
legLabels  = [legLabels; {'Moon'; 'L1'; 'L2'}];

lgd = legend(ax, legHandles, legLabels, 'Location','northeast');
lgd.Box = 'on';
lgd.ItemTokenSize = [18 12];
if numel(legLabels) > 6
    lgd.NumColumns = 2;
end

lgd.Units = 'normalized';
pos = lgd.Position;
pos(1) = pos(1) + 0.075;
pos(2) = pos(2) + 0.04;
lgd.Position = pos;

axis(ax,'equal');
axis(ax,'tight');
ax.Units = 'normalized';
ax.PositionConstraint = 'innerposition';

pad = 0.10;
ax.Position = [pad pad 1-2*pad 1-2*pad];

ax.LooseInset = ax.TightInset + [0.02 0.02 0.02 0.02];
axis(ax,'vis3d');

exportgraphics(fig, fullfile(FigDir, sprintf('fig_traj3d_%s.pdf', FILE_TAG)), 'ContentType','image');
savefig(fig, fullfile(FigDir, sprintf('fig_traj3d_%s.fig', FILE_TAG)));

% ---------------- 3-sigma plots ----------------
Nf = size(cov,1);
sig = zeros(Nf,6);
for k = 1:Nf
    Pk = squeeze(cov(k,:,:));
    sig(k,:) = sqrt(max(diag(Pk),0));
end
sig3 = 3*sig;

t = t_target_ekf(:);
err = s_ekf(:,1:6) - s_target_ekf(:,1:6);
err_pos_km   = err(:,1:3) * LU;
err_vel_kms  = err(:,4:6) * VU;
sig3_pos_km  = sig3(:,1:3) * LU;
sig3_vel_kms = sig3(:,4:6) * VU;

cBound = [0.85 0.10 0.10];
cErr   = [0.00 0.45 0.74];

yLbls = { ...
    'e_x (km)', ...
    'e_y (km)', ...
    'e_z (km)', ...
    'e_{v_x} (km/s)', ...
    'e_{v_y} (km/s)', ...
    'e_{v_z} (km/s)'};

err_all  = [err_pos_km,  err_vel_kms];
sig3_all = [sig3_pos_km, sig3_vel_kms];

plotSigFig = @(fName, xData, errData, sigData, yLbl) ...
    create_sig_fig(fName, xData, errData, sigData, yLbl, figW, figH, ...
                   cBound, cErr, FigDir, availableObsCount, num_obs);

plotSigFig(sprintf('fig_3sig_x_%s.pdf',  FILE_TAG), t, err_pos_km(:,1),  sig3_pos_km(:,1),  'e_x (km)');
plotSigFig(sprintf('fig_3sig_y_%s.pdf',  FILE_TAG), t, err_pos_km(:,2),  sig3_pos_km(:,2),  'e_y (km)');
plotSigFig(sprintf('fig_3sig_z_%s.pdf',  FILE_TAG), t, err_pos_km(:,3),  sig3_pos_km(:,3),  'e_z (km)');
plotSigFig(sprintf('fig_3sig_vx_%s.pdf', FILE_TAG), t, err_vel_kms(:,1), sig3_vel_kms(:,1), 'e_{v_x} (km/s)');
plotSigFig(sprintf('fig_3sig_vy_%s.pdf', FILE_TAG), t, err_vel_kms(:,2), sig3_vel_kms(:,2), 'e_{v_y} (km/s)');
plotSigFig(sprintf('fig_3sig_vz_%s.pdf', FILE_TAG), t, err_vel_kms(:,3), sig3_vel_kms(:,3), 'e_{v_z} (km/s)');

gridFigW = 12;
gridFigH = 6.8;
create_sig_fig_grid(sprintf('fig_3sig_grid_%s.pdf', FILE_TAG), t, err_all, sig3_all, yLbls, ...
    gridFigW, gridFigH, cBound, cErr, FigDir, availableObsCount, num_obs);

% ---------------- EKF performance print statements ----------------
rmse_pos = sqrt(mean(sum((s_ekf(:,1:3) - s_target_ekf(:,1:3)).^2,2)));
rmse_vel = sqrt(mean(sum((s_ekf(:,4:6) - s_target_ekf(:,4:6)).^2,2)));

rmse_pos_km  = rmse_pos * LU;
rmse_vel_kms = rmse_vel * VU;

detPpos = zeros(Nf,1);
for k = 1:Nf
    Pk = squeeze(cov(k,:,:));
    Ppos = Pk(1:3,1:3);
    detPpos(k) = det(Ppos);
end
detPpos_km6 = detPpos * (LU^6);

mean_stability = mean(stabilities(orbit_indices));

safe_printf('\n--- EKF PERFORMANCE ---\n');
safe_printf('RMSE position (km):     %.6e\n', rmse_pos_km);
safe_printf('RMSE velocity (km/s):   %.6e\n', rmse_vel_kms);
safe_printf('Mean det(P_pos) (km^6): %.6e\n', mean(detPpos_km6));
safe_printf('Mean stability:         %.6e\n', mean_stability);

% ---------------- One Excel file ----------------
try
    if exist('min_cost','var')
        minCostVal = min_cost;
    else
        minCostVal = NaN;
    end

    summaryRow = table( ...
        string(RUN_TAG), string(OPTIMIZER_MODE), seedVal, ...
        logical(useScreening), logical(costFlags.J1), logical(costFlags.J2), logical(costFlags.J3), ...
        MAX_ITERS, MAX_EVALS, TotalRuntime, screeningCount_final, ...
        rmse_pos_km, rmse_vel_kms, mean(detPpos_km6), mean_stability, minCostVal, ...
        'VariableNames', { ...
            'run_tag','optimizer','seed', ...
            'use_screening','use_J1','use_J2','use_J3', ...
            'max_iters','max_evals','runtime_s','screeningCount_final', ...
            'rmse_pos_km','rmse_vel_kms','mean_detPpos_km6','mean_stability','min_cost' ...
        });

    if isfile(EXCEL_FILE)
        writetable(summaryRow, EXCEL_FILE, 'Sheet','Summary', 'WriteMode','append');
    else
        writetable(summaryRow, EXCEL_FILE, 'Sheet','Summary');
    end

    logCell = evalin('base','OptimizationLog');
    if iscell(logCell) && ~isempty(logCell)
        logStruct = vertcat(logCell{:});
        histTbl = struct2table(logStruct, "AsArray", true);
    else
        histTbl = table();
    end

    sheetName = matlab.lang.makeValidName(RUN_TAG);
    sheetName = replace(sheetName,"_","");
    if strlength(sheetName) > 31
        sheetName = extractBefore(sheetName, 32);
    end
    writetable(histTbl, EXCEL_FILE, 'Sheet', char(sheetName));

    obsSheet = matlab.lang.makeValidName(RUN_TAG + "_obs");
    obsSheet = replace(obsSheet,"_","");
    if strlength(obsSheet) > 31
        obsSheet = extractBefore(obsSheet, 32);
    end
    writetable(obsTbl, EXCEL_FILE, 'Sheet', char(obsSheet));

catch ME
    safe_printf(2,"WARNING: failed to write ExperimentSummary file: %s\n", ME.message);
end

try
    diary off
catch
end

% ---------------- Helper Functions ----------------
function cacheKey = make_transfer_cache_key(missionCfg, slots_per_orbit)
    tr = missionCfg.transfer;
    solverMode = upper(string(tr.solverMode));

    depOrb = get_field_or_default(tr, 'depOrbitIndex', 0);
    arrOrb = get_field_or_default(tr, 'arrOrbitIndex', 0);
    depSlot = get_field_or_default(tr, 'depSlot', 0);
    arrSlot = get_field_or_default(tr, 'arrSlot', 0);
    dtVal = get_field_or_default(tr, 'dt', 0);

    switch solverMode
        case "LOW_THRUST_CLASS"
            lt = tr.lowthrust;

            cacheKey = sprintf('lt_d%d_a%d_ds%d_as%d_dt%s_sl%d_tf%s', ...
                depOrb, ...
                arrOrb, ...
                depSlot, ...
                arrSlot, ...
                local_num_str(dtVal), ...
                slots_per_orbit, ...
                local_num_str(get_field_or_default(lt, 'tf_guess', 0)));

        otherwise
            cacheKey = sprintf('tr_d%d_a%d_ds%d_as%d_dt%s_sl%d', ...
                depOrb, arrOrb, depSlot, arrSlot, local_num_str(dtVal), slots_per_orbit);
    end

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

function s = local_vec_str(v) 
    v = v(:).';
    c = strings(1, numel(v));
    for i = 1:numel(v)
        c(i) = string(sprintf('%.6g', v(i)));
    end
    s = strjoin(c, "_");
end

function obsCount = sanitize_obs_count_vector(obsCount, N, num_obs)
    if isempty(obsCount)
        obsCount = NaN(N,1);
        return;
    end

    obsCount = obsCount(:);
    if numel(obsCount) ~= N
        obsCount = NaN(N,1);
        return;
    end

    obsCount = round(obsCount);
    obsCount = max(0, min(num_obs, obsCount));
end

function cmap = make_obscount_cmap(maxObs)
    maxObs = max(0, round(maxObs));
    nBands = maxObs + 1;

    if nBands == 1
        grayVals = 1.0;
    else
        grayVals = linspace(0.40, 1.00, nBands).';
    end

    cmap = [grayVals grayVals grayVals];
end

function create_sig_fig(fName, t, err, sig3, yLbl, w, h, cBnd, cErr, outDir, obsCount, maxObs)
    make_one_sig_fig( ...
        fullfile(outDir, fName), ...
        t, err, sig3, yLbl, w, h, cBnd, cErr, obsCount, maxObs, false);

    [~, base, ext] = fileparts(fName);
    fNameZoom = fullfile(outDir, base + "_zoom" + ext);

    make_one_sig_fig( ...
        fNameZoom, ...
        t, err, sig3, yLbl, w, h, cBnd, cErr, obsCount, maxObs, true);
end

function make_one_sig_fig(savePath, t, err, sig3, yLbl, w, h, cBnd, cErr, obsCount, maxObs, doZoom)
    f = figure('Color','w','Units','inches','Position',[1 1 w h], ...
               'PaperUnits','inches','PaperPosition',[0 0 w h]);
    ax = axes(f);
    hold(ax,'on');
    box(ax,'on');
    set(ax,'TickLabelInterpreter','tex', 'Layer','top');

    yMax = max(abs([err(:); sig3(:)]));
    if doZoom
        vals = abs([err(:); sig3(:)]);
        vals = vals(isfinite(vals));
        if ~isempty(vals)
            yMax = prctile(vals, 98);
        end
    end

    if ~isfinite(yMax) || yMax <= 0
        yMax = 1;
    end
    yPad = 0.08 * yMax;
    yLims = [-yMax-yPad, yMax+yPad];

    if ~isempty(obsCount) && ~all(isnan(obsCount))
        obsCount = obsCount(:).';
        bg = repmat(obsCount, 2, 1);

        imagesc(ax, t(:).', yLims, bg);
        set(ax, 'YDir', 'normal');

        cmap = make_obscount_cmap(maxObs);
        colormap(ax, cmap);
        clim(ax, [-0.5, maxObs + 0.5]);
    end

    hB = plot(ax, t,  sig3, '-', 'Color', cBnd);
         plot(ax, t, -sig3, '-', 'Color', cBnd);
    hE = plot(ax, t,  err,  '-', 'Color', cErr);

    xlabel(ax, 't (TU)');
    ylabel(ax, yLbl);
    xlim(ax, [t(1) t(end)]);
    ylim(ax, yLims);

    lgd = legend(ax, [hE, hB], {'EKF error', '\pm 3\sigma bound'}, ...
        'Location', 'northeast');
    lgd.Box = 'on';
    lgd.ItemTokenSize = [18 12];

    if ~isempty(obsCount) && ~all(isnan(obsCount))
        cb = colorbar(ax);
        cb.Location = 'eastoutside';
        cb.Label.String = 'Available observers';
        cb.Ticks = 0:maxObs;
        cb.TickLabels = string(0:maxObs);
        cb.TickDirection = 'out';

        ax.Units = 'normalized';
        ax.Position = [0.12 0.14 0.68 0.80];

        cb.Units = 'normalized';
        cb.Position = [0.84 0.14 0.03 0.80];
        pos = cb.Position;
        pos(1) = pos(1) + 0.01;
        cb.Position = pos;
    end

    exportgraphics(f, savePath, 'ContentType','image');
    savefig(f, replace(savePath, '.pdf', '.fig'));
    close(f);
end

function create_sig_fig_grid(fName, t, errMat, sig3Mat, yLbls, w, h, cBnd, cErr, outDir, obsCount, maxObs)
    make_one_sig_grid_fig( ...
        fullfile(outDir, fName), ...
        t, errMat, sig3Mat, yLbls, w, h, cBnd, cErr, obsCount, maxObs, false);

    [~, base, ext] = fileparts(fName);
    fNameZoom = fullfile(outDir, base + "_zoom" + ext);

    make_one_sig_grid_fig( ...
        fNameZoom, ...
        t, errMat, sig3Mat, yLbls, w, h, cBnd, cErr, obsCount, maxObs, true);
end

function make_one_sig_grid_fig(savePath, t, errMat, sig3Mat, yLbls, w, h, cBnd, cErr, obsCount, maxObs, doZoom)
    f = figure('Color','w','Units','inches','Position',[1 1 w h], ...
               'PaperUnits','inches','PaperPosition',[0 0 w h]);

    tl = tiledlayout(f, 2, 3, 'TileSpacing','compact', 'Padding','compact');

    ax = gobjects(6,1);
    hE = gobjects(6,1);
    hB = gobjects(6,1);

    for i = 1:6
        ax(i) = nexttile(tl);
        hold(ax(i),'on');
        box(ax(i),'on');
        set(ax(i),'TickLabelInterpreter','tex', 'Layer','top');

        err  = errMat(:,i);
        sig3 = sig3Mat(:,i);

        yMax = max(abs([err(:); sig3(:)]));
        if doZoom
            vals = abs([err(:); sig3(:)]);
            vals = vals(isfinite(vals));
            if ~isempty(vals)
                yMax = prctile(vals, 98);
            end
        end

        if ~isfinite(yMax) || yMax <= 0
            yMax = 1;
        end
        yPad = 0.08 * yMax;
        yLims = [-yMax-yPad, yMax+yPad];

        if ~isempty(obsCount) && ~all(isnan(obsCount))
            obsCountRow = obsCount(:).';
            bg = repmat(obsCountRow, 2, 1);

            imagesc(ax(i), t(:).', yLims, bg);
            set(ax(i), 'YDir', 'normal');

            cmap = make_obscount_cmap(maxObs);
            colormap(ax(i), cmap);
            clim(ax(i), [-0.5, maxObs + 0.5]);
        end

        hB(i) = plot(ax(i), t,  sig3, '-', 'Color', cBnd);
                  plot(ax(i), t, -sig3, '-', 'Color', cBnd);
        hE(i) = plot(ax(i), t,  err,  '-', 'Color', cErr);

        ylabel(ax(i), yLbls{i});
        xlim(ax(i), [t(1) t(end)]);
        ylim(ax(i), yLims);

        if i > 3
            xlabel(ax(i), 't (TU)');
        end
    end

    lgd = legend(ax(1), [hE(1), hB(1)], {'EKF error', '\pm 3\sigma bound'}, ...
        'Location', 'northeast');
    lgd.Box = 'on';
    lgd.ItemTokenSize = [18 12];

    if ~isempty(obsCount) && ~all(isnan(obsCount))
        cb = colorbar(ax(6));
        cb.Label.String = 'Available observers';
        cb.Ticks = 0:maxObs;
        cb.TickLabels = string(0:maxObs);
        cb.TickDirection = 'out';

        try
            cb.Layout.Tile = 'east';
        catch
            cb.Units = 'normalized';
            cb.Position = [0.92 0.14 0.02 0.76];
        end
    end

    exportgraphics(f, savePath, 'ContentType','image');
    savefig(f, replace(savePath, '.pdf', '.fig'));
    close(f);
end

function [xL1, xL2] = cr3bp_L1L2(mu)
    f = @(x) x ...
        - (1-mu)*(x + mu)./abs(x + mu).^3 ...
        - mu*(x - (1-mu))./abs(x - (1-mu)).^3;

    delta = (mu/3)^(1/3);
    x2 = 1 - mu;

    x0_L1 = x2 - delta;
    x0_L2 = x2 + delta;

    opts = optimset('Display','off');
    xL1 = fzero(f, x0_L1, opts);
    xL2 = fzero(f, x0_L2, opts);
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

function safe_disp(x)
    try
        if ischar(x) || isstring(x)
            msg = char(string(x));
        else
            msg = evalc('disp(x)');
        end

        try
            fprintf('%s', msg);
            if isempty(msg) || msg(end) ~= newline
                fprintf('\n');
            end
        catch
        end

        append_fallback_output(msg);
    catch
    end
end

function append_fallback_output(msg)
    try
        if isstring(msg)
            msg = char(msg);
        end

        fallbackFile = getenv('SAFE_FALLBACK_FILE');
        if isempty(fallbackFile)
            fallbackFile = fullfile(pwd, 'safe_output_fallback.txt');
        end

        fid = fopen(fallbackFile, 'a');
        if fid ~= -1
            fprintf(fid, '%s', msg);
            if isempty(msg) || msg(end) ~= newline
                fprintf(fid, '\n');
            end
            fclose(fid);
        end
    catch
    end
end

function [state, options, optchanged] = ga_outfun(options, state, flag)
    optchanged = false;
    try
        if strcmp(flag, 'iter')
            if ~isempty(state.Score)
                bestScore = min(state.Score);
            else
                bestScore = NaN;
            end
            safe_printf('GA gen %3d | bestJ = %.12g\n', state.Generation, bestScore);
        elseif strcmp(flag, 'done')
            safe_printf('GA finished.\n');
        end
    catch
    end
end

function stop = pso_outfun(optimValues, state)
    stop = false;
    try
        if strcmp(state, 'iter')
            safe_printf('PSO iter %3d | bestJ = %.12g\n', ...
                optimValues.iteration, optimValues.bestfval);
        elseif strcmp(state, 'done')
            safe_printf('PSO finished.\n');
        end
    catch
    end
end

function J = bayes_objective_with_logging(T)
    ObjFcn = getappdata(0, 'BAYES_OBJFCN');

    bayesEvalCounter = getappdata(0, 'BAYES_EVAL_COUNTER');
    bayesBestCost    = getappdata(0, 'BAYES_BEST_COST');

    x = table2array(T);
    J = ObjFcn(x);

    bayesEvalCounter = bayesEvalCounter + 1;
    if J < bayesBestCost
        bayesBestCost = J;
    end

    setappdata(0, 'BAYES_EVAL_COUNTER', bayesEvalCounter);
    setappdata(0, 'BAYES_BEST_COST', bayesBestCost);

    safe_printf('BAYES eval %3d | J = %.12g | bestJ = %.12g\n', ...
        bayesEvalCounter, J, bayesBestCost);
end