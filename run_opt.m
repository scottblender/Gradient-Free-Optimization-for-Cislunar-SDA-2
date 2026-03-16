% ---- run_optimization.m ---- %
clear; close all; clc;

fprintf('RUN START: %s\n', string(datetime('now')));
drawnow;

% defaults for figures (bold)
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

% load in filtered and sorted JPL data
S  = load('JPL_CR3BP_OrbitCatalog.mat');
CatalogDir = pwd;
T1 = S.T;
t_lg = S.t_lg;
s_lg = S.s_lg;

% User-specified Inputs
% Options: 'GA', 'PSO', 'BAYESIAN', 'GAMULTIOBJ', 'DMOPSO', 'ABC', 'ACO'
OPTIMIZER_MODE = 'GA'; % default

envMode = getenv("OPTIMIZER_MODE");
if ~isempty(envMode)
    OPTIMIZER_MODE = envMode;
end
OPTIMIZER_MODE = upper(string(OPTIMIZER_MODE));

% Stopping Criteria (max iterations for all except Bayesian)
MAX_ITERS = 5;
MAX_EVALS = 100;  % Bayesian only (objective evaluation budget)

% JPL Constants
mu = 1.215058560962404E-2;
LU = 384400;     % km
TU = 375695;     % seconds
VU = LU / TU;    % km/s

ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% --- EKF Parameters ---
pos_var  = (1 / LU)^2;
vel_var  = (10 / (VU * 1000))^2;
P_0_base = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);
Q_k      = diag(repmat(1e-8, 6, 1));
R_k_base = diag([1e-8, 1e-8]);

% --- Lunar Gateway ICs ---
s_lg_ic     = [1.02202108343387, 0, -0.182096487798513, 0, -0.103255420206012, 0]';
tspan_lg_ic = [0, 1.51110546287394];

% MILP-Implementation
num_orbits      = height(T1); % number of candidate orbits
slots_per_orbit = 50;         % number of discrete slots per orbit

tf          = T1.("Period (TU) ");
states      = T1.("state");
times       = T1.("time");
stabilities = T1.("Stability index  ");

% shared cache folder for orbit slot database
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
            fprintf('Loaded cached orbit database from:\n  %s\n', orbitDbCacheFile);
        end
    catch ME
        fprintf(2, 'WARNING: failed to load orbit database cache: %s\n', ME.message);
        rebuildOrbitDb = true;
    end
end

if rebuildOrbitDb
    fprintf('Building orbit database for %d slots/orbit...\n', slots_per_orbit);

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
        fprintf('Saved orbit database cache to:\n  %s\n', orbitDbCacheFile);
    catch ME
        fprintf(2, 'WARNING: failed to save orbit database cache: %s\n', ME.message);
    end
end

% artifacts folders + console capture
runDirEnv = getenv("RUN_DIR");
if ~isempty(runDirEnv)
    if ~exist(runDirEnv,'dir')
        mkdir(runDirEnv);
    end
    cd(runDirEnv);
end

RunDir = pwd;

% Base artifacts folder (stays under RUN_DIR)
ArtDir  = fullfile(RunDir, "artifacts");
if ~exist(ArtDir,'dir'),  mkdir(ArtDir);  end

% Shared transfer truth cache folder
TransferCacheDir = fullfile(RunDir, "transfer_cache");
if ~exist(TransferCacheDir,'dir'), mkdir(TransferCacheDir); end

% ---- Visibility Parameters ----
sun_min_deg  = 20;   % Sun exclusion angle (deg)
moon_min_deg = 10;   % Moon exclusion angle (deg)

sun_min  = deg2rad(sun_min_deg);
moon_min = deg2rad(moon_min_deg);

theta0 = 0;             % initial phase angle (rad)
i_sun  = deg2rad(0);    % keep planar for now

sunFcn = @(t) sun_pos_bc4bp(t, LU, TU, theta0, i_sun);

% choose whether or not to include occlusion/exclusion
useScreening = true;

% struct to include or exclude cost components
costFlags = struct('J1', true, 'J2', true, 'J3', true);

% if using powershell script
v = getenv("MAX_ITERS"); if ~isempty(v), MAX_ITERS = str2double(v); end

v = getenv("USE_SCREENING");
if ~isempty(v), useScreening = (str2double(v) ~= 0); end

vj1 = getenv("USE_J1"); if ~isempty(vj1), costFlags.J1 = (str2double(vj1) ~= 0); end
vj2 = getenv("USE_J2"); if ~isempty(vj2), costFlags.J2 = (str2double(vj2) ~= 0); end
vj3 = getenv("USE_J3"); if ~isempty(vj3), costFlags.J3 = (str2double(vj3) ~= 0); end

% --- run tag + per-run folders + one excel file ---
vseed = getenv("SEED"); if isempty(vseed), vseed = "0"; end
seedVal = str2double(vseed); if isnan(seedVal), seedVal = 0; end

RUN_TAG = sprintf('%s_scr%d_J%d%d%d_seed%03d', char(OPTIMIZER_MODE), ...
    double(useScreening), double(costFlags.J1), double(costFlags.J2), double(costFlags.J3), seedVal);

RunArtDir  = fullfile(ArtDir, RUN_TAG);
FigDir     = fullfile(RunArtDir, "figs");
DataDir    = fullfile(RunArtDir, "data");
LogDir     = fullfile(RunArtDir, "logs");
if ~exist(RunArtDir,'dir'), mkdir(RunArtDir); end
if ~exist(FigDir,'dir'),    mkdir(FigDir);    end
if ~exist(DataDir,'dir'),   mkdir(DataDir);   end
if ~exist(LogDir,'dir'),    mkdir(LogDir);    end

EXCEL_FILE = fullfile(DataDir, "ExperimentSummary.xlsx");

try
    diaryFile = fullfile(LogDir, "matlab_diary_" + string(datetime("now","Format","yyyyMMdd_HHmmss")) + ".txt");
    diary(diaryFile);
    diary on
catch
end

% set up data logging (in-memory only)
dq = parallel.pool.DataQueue;
assignin('base', 'OptimizationLog', {});
afterEach(dq, @(data) append_log(data));

function append_log(data)
    logCell = evalin('base', 'OptimizationLog');
    logCell{end+1,1} = data;
    assignin('base', 'OptimizationLog', logCell);
end

% set flag for single or multi-objective
opt_flag          = 'SOO';
const_stabilities = parallel.pool.Constant(stabilities);
const_orbit_db    = parallel.pool.Constant(orbit_database); %#ok<NASGU>

% ---------------- Mission type ----------------
MISSION_TYPE = 'LOW_THRUST_TRANSFER';

envMission = getenv("MISSION_TYPE");
if ~isempty(envMission)
    MISSION_TYPE = upper(string(envMission));
end

% ---------------- Mission config ----------------
missionCfg = struct();
missionCfg.type = upper(string(MISSION_TYPE));

% NEW: set observer count once here; all optimizers use it
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

    case "BALLISTIC_TRANSFER"
        missionCfg.transfer.depOrbitIndex = 52;
        missionCfg.transfer.depSlot       = 10;
        missionCfg.transfer.arrOrbitIndex = 400;
        missionCfg.transfer.dt            = 0.001;
        missionCfg.transfer.solverMode    = "BALLISTIC";

        missionCfg.transfer.ballistic.dv_guess    = [0;0;0];
        missionCfg.transfer.ballistic.tf_guess    = 2.0;
        missionCfg.transfer.ballistic.phase_guess = 0.5 * T1.("Period (TU) ")(missionCfg.transfer.arrOrbitIndex);
        missionCfg.transfer.ballistic.dv_max      = 0.05;
        missionCfg.transfer.ballistic.tf_lb       = 0.1;
        missionCfg.transfer.ballistic.tf_ub       = 10.0;

    case "LOW_THRUST_TRANSFER"
        missionCfg.transfer.depOrbitIndex = 52;
        missionCfg.transfer.depSlot       = 10;
        missionCfg.transfer.arrOrbitIndex = 400;
        missionCfg.transfer.dt            = 0.001;
        missionCfg.transfer.solverMode    = "LOW_THRUST";

        missionCfg.transfer.lowthrust.Nseg        = 40;
        missionCfg.transfer.lowthrust.m0          = 1.0;
        missionCfg.transfer.lowthrust.Tmax        = 0.3672;
        missionCfg.transfer.lowthrust.ve          = 10.0;
        missionCfg.transfer.lowthrust.tf_guess    = 2.0;
        missionCfg.transfer.lowthrust.tf_lb       = 0.1;
        missionCfg.transfer.lowthrust.tf_ub       = 12.0;
        missionCfg.transfer.lowthrust.phase_guess = 0.4;

        missionCfg.transfer.lowthrust.w_pos       = 1e4;
        missionCfg.transfer.lowthrust.w_vel       = 1e3;
        missionCfg.transfer.lowthrust.w_tf        = 1e-2;
        missionCfg.transfer.lowthrust.w_smooth    = 1e-2;
        missionCfg.transfer.lowthrust.w_smooth2   = 1e-1;
        missionCfg.transfer.lowthrust.w_ctrl      = 1e-4;

    otherwise
        error("Unknown MISSION_TYPE: %s", missionCfg.type);
end

% NEW: dynamic optimizer sizing
num_obs_cfg = missionCfg.optimization.numObservers;
nVars_common = 2 * num_obs_cfg;
LB_common = repmat([1, 1], 1, num_obs_cfg);
UB_common = repmat([num_orbits, slots_per_orbit], 1, num_obs_cfg);

% ---------------- Build/load target truth ----------------
useTransferCache = true;

if contains(string(missionCfg.type), "TRANSFER") && useTransferCache
    cacheKey  = make_transfer_cache_key(missionCfg, slots_per_orbit);
    cacheFile = fullfile(TransferCacheDir, cacheKey + ".mat");

    if isfile(cacheFile)
        fprintf('Loading cached transfer truth from:\n  %s\n', cacheFile);
        C = load(cacheFile, 't_target', 's_target', 'truthInfo', 'cacheMeta');
        t_target  = C.t_target;
        s_target  = C.s_target;
        truthInfo = C.truthInfo;

        if isfield(C, 'cacheMeta')
            fprintf('Cached transfer key: %s\n', string(C.cacheMeta.cacheKey));
        end
    else
        fprintf('No cached transfer found. Computing transfer truth...\n');

        [t_target, s_target, truthInfo] = build_target_truth( ...
            missionCfg, T1, orbit_database, times, states, mu, ode_opts);

        cacheMeta = struct();
        cacheMeta.cacheKey        = cacheKey;
        cacheMeta.missionType     = string(missionCfg.type);
        cacheMeta.created         = string(datetime('now'));
        cacheMeta.slots_per_orbit = slots_per_orbit;
        cacheMeta.mu              = mu;

        try
            save(cacheFile, 't_target', 's_target', 'truthInfo', 'cacheMeta', '-v7.3');
            fprintf('Saved transfer truth cache to:\n  %s\n', cacheFile);
        catch ME
            fprintf(2, 'WARNING: failed to save transfer cache: %s\n', ME.message);
        end
    end
else
    [t_target, s_target, truthInfo] = build_target_truth( ...
        missionCfg, T1, orbit_database, times, states, mu, ode_opts);
end

disp('--- Truth Info ---');
disp(truthInfo);

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
ObjFcn = @(x) objective_wrapper(x, orbit_database, stabilities, s_target_ekf, t_target_ekf, P_0_base, Q_k, R_k_base, mu, LU, ...
    sunFcn, sun_min, moon_min, opt_flag, OPTIMIZER_MODE, dq, useScreening, costFlags);

RunTimer = tic;

switch upper(OPTIMIZER_MODE)

    case 'GA'
        fprintf('Starting Genetic Algorithm...\n');
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
            'FitnessLimit', -Inf);

        [x_best, min_cost, exitflag, output, population, scores] = ga(ObjFcn, nVars, [], [], [], [], LB, UB, [], IntCon, options);
        J_check = ObjFcn(x_best);

        fprintf('ga returned min_cost = %.12f\n', min_cost);
        fprintf('reevaluated J(x_best) = %.12f\n', J_check);

        [bestFinalScore, idxBestFinal] = min(scores);
        fprintf('best score in final population = %.12f\n', bestFinalScore);
        disp('x_best returned by ga:');
        disp(x_best)
        disp('best individual in final population:');
        disp(population(idxBestFinal,:))

    case 'PSO'
        fprintf('Starting Particle Swarm Optimization...\n');
        nVars = nVars_common;
        LB = LB_common;
        UB = UB_common;

        swarm = 60;

        options = optimoptions('particleswarm', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'SwarmSize', swarm, ...
            'MaxIterations', MAX_ITERS);

        [x_best, min_cost] = particleswarm(ObjFcn, nVars, LB, UB, options);
        x_best = round(x_best);

    case 'BAYESIAN'
        fprintf('Starting Bayesian Optimization...\n');

        vars = [];
        for i = 1:num_obs_cfg
            vars = [vars, ...
                optimizableVariable(['Orbit',num2str(i)], [1, num_orbits], 'Type','integer'), ...
                optimizableVariable(['Slot', num2str(i)], [1, slots_per_orbit], 'Type','integer')]; %#ok<AGROW>
        end

        results = bayesopt(ObjFcn, vars, ...
            'UseParallel', true, ...
            'IsObjectiveDeterministic', false, ...
            'MaxObjectiveEvaluations', MAX_EVALS);

        x_best   = table2array(results.XAtMinObjective);
        min_cost = results.MinObjective;

    case 'GAMULTIOBJ'
        fprintf('Starting Multi-Objective Genetic Algorithm (NSGA-II)...\n');

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
        fprintf('Starting Custom Multi-Objective PSO...\n');

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
        fprintf('Starting Artificial Bee Colony Optimization...\n');

        LB = LB_common;
        UB = UB_common;

        abc_opts.ColonySize      = 60;
        abc_opts.MaxIters        = MAX_ITERS;
        abc_opts.Limit           = 20;
        abc_opts.StallIters      = inf;
        abc_opts.SlotsPerOrbit   = slots_per_orbit;
        abc_opts.UseParallel     = true;
        abc_opts.UseParallelInit = true;

        [x_best, min_cost] = abc_discrete(ObjFcn, LB, UB, abc_opts);

    case 'ACO'
        fprintf('Starting Ant Colony Optimization...\n');

        LB = LB_common;
        UB = UB_common;

        aco_opts.nAnts       = 60;
        aco_opts.MaxIters    = MAX_ITERS;
        aco_opts.alpha       = 1.0;
        aco_opts.beta        = 1.0;
        aco_opts.rho         = 0.2;
        aco_opts.Q           = 1.0;
        aco_opts.UseParallel = true;
        aco_opts.TauMin              = 1e-12;
        aco_opts.UseIterBestDeposit  = true;
        aco_opts.IterBestWeight      = 1.0;
        aco_opts.StallIters          = inf;

        [x_best, min_cost] = aco_discrete(ObjFcn, LB, UB, aco_opts);

    otherwise
        error("Unknown OPTIMIZER_MODE: %s", OPTIMIZER_MODE);
end

% runtime
TotalRuntime = toc(RunTimer);
fprintf('Total Runtime: %.2f seconds\n', TotalRuntime);

% print results
if strcmpi(opt_flag, 'SOO')
    fprintf('\n--- FINAL RESULTS (%s) ---\n', OPTIMIZER_MODE);
    fprintf('Orbits: %s\n', mat2str(x_best(1:2:end)));
    fprintf('Slots:  %s\n', mat2str(x_best(2:2:end)));
    fprintf('Cost:   %.4f\n', min_cost);
    x_plot = x_best;
else
    f_min  = min(fval);
    f_max  = max(fval);
    f_norm = (fval - f_min) ./ (f_max - f_min);

    dist_to_utopia = sqrt(sum(f_norm.^2, 2));
    [~, idx_knee]  = min(dist_to_utopia);

    knee_costs = fval(idx_knee, :);
    knee_vars  = x_best(idx_knee, :);

    fprintf('\n--- KNEE POINT (Balanced Solution) ---\n');
    fprintf('Selected Row: %d\n', idx_knee);
    fprintf('RMSE (Log):   %.4f\n', knee_costs(1));
    fprintf('Det (Log):    %.4f\n', knee_costs(2));
    fprintf('Stability:    %.4f\n', knee_costs(3));
    fprintf('Orbits:       %s\n', mat2str(knee_vars(1:2:end)));
    fprintf('Slots:        %s\n', mat2str(knee_vars(2:2:end)));
    x_plot = knee_vars;
end

fprintf('RUN END: %s\n', string(datetime('now')));
drawnow;

% --- parallel pool cleanup ---
try
    p = gcp('nocreate');
    if ~isempty(p)
        delete(p);
    end
catch
end

drawnow;
pause(0.2);

% ---------------- Recompile results to plot  ----------------
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

% NEW: try to get per-step available observer counts from EKF if supported
availableObsCount = [];
try
    [s_ekf, cov, screeningCount_final, availableObsCount] = cr3bp_ekf(observer_ICs, s_target_ekf, t_target_ekf, ...
        P_0_base, Q_k, R_k_base, mu, LU, sunFcn, sun_min, moon_min, useScreening);
catch
    [s_ekf, cov, screeningCount_final] = cr3bp_ekf(observer_ICs, s_target_ekf, t_target_ekf, ...
        P_0_base, Q_k, R_k_base, mu, LU, sunFcn, sun_min, moon_min, useScreening);
end

fprintf('\nFinal EKF screeningCount = %d\n', screeningCount_final);

availableObsCount = sanitize_obs_count_vector(availableObsCount, numel(t_target_ekf), num_obs);

% ---------------- observer metadata ----------------
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

% ---------------- trajectory plot ----------------
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

hEKF = plot3(ax, s_ekf(:,1), s_ekf(:,2), s_ekf(:,3), '-', ...
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

    hStart = plot3(ax, s_target_ekf(1,1), s_target_ekf(1,2), s_target_ekf(1,3), 'o', ...
        'MarkerSize', 9, 'MarkerFaceColor', depBase, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.0);

    hEnd = plot3(ax, s_target_ekf(end,1), s_target_ekf(end,2), s_target_ekf(end,3), 's', ...
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

plot3(ax, s_target_ekf(1,1), s_target_ekf(1,2), s_target_ekf(1,3), 'o', ...
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

exportgraphics(fig, fullfile(FigDir,'fig_traj3d.pdf'), 'ContentType','image');
savefig(fig, fullfile(FigDir,'fig_traj3d.fig'));   % NEW

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
err_pos_km  = err(:,1:3) * LU;
err_vel_kms = err(:,4:6) * VU;
sig3_pos_km  = sig3(:,1:3) * LU;
sig3_vel_kms = sig3(:,4:6) * VU;
cBound = [0.85 0.10 0.10];
cErr   = [0.00 0.45 0.74];

plotSigFig = @(fName, xData, errData, sigData, yLbl) ...
    create_sig_fig(fName, xData, errData, sigData, yLbl, figW, figH, ...
                   cBound, cErr, FigDir, availableObsCount, num_obs);

plotSigFig('fig_3sig_x.pdf', t, err_pos_km(:,1), sig3_pos_km(:,1), 'e_x (km)');
plotSigFig('fig_3sig_y.pdf', t, err_pos_km(:,2), sig3_pos_km(:,2), 'e_y (km)');
plotSigFig('fig_3sig_z.pdf', t, err_pos_km(:,3), sig3_pos_km(:,3), 'e_z (km)');
plotSigFig('fig_3sig_vx.pdf', t, err_vel_kms(:,1), sig3_vel_kms(:,1), 'e_{v_x} (km/s)');
plotSigFig('fig_3sig_vy.pdf', t, err_vel_kms(:,2), sig3_vel_kms(:,2), 'e_{v_y} (km/s)');
plotSigFig('fig_3sig_vz.pdf', t, err_vel_kms(:,3), sig3_vel_kms(:,3), 'e_{v_z} (km/s)');

% ---------------- print statements ----------------
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

fprintf('\n--- EKF PERFORMANCE ---\n');
fprintf('RMSE position (km):     %.6e\n', rmse_pos_km);
fprintf('RMSE velocity (km/s):   %.6e\n', rmse_vel_kms);
fprintf('Mean det(P_pos) (km^6): %.6e\n', mean(detPpos_km6));

% --- one Excel file ---
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
        rmse_pos_km, rmse_vel_kms, mean(detPpos_km6), minCostVal, ...
        'VariableNames', { ...
            'run_tag','optimizer','seed', ...
            'use_screening','use_J1','use_J2','use_J3', ...
            'max_iters','max_evals','runtime_s','screeningCount_final', ...
            'rmse_pos_km','rmse_vel_kms','mean_detPpos_km6','min_cost' ...
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
    fprintf(2,"WARNING: failed to write ExperimentSummary.xlsx: %s\n", ME.message);
end

diary off

% ---------------- Helper Functions ----------------
function cacheKey = make_transfer_cache_key(missionCfg, slots_per_orbit)
    tr = missionCfg.transfer;
    solverMode = upper(string(tr.solverMode));

    parts = strings(0,1);
    parts(end+1) = "type_" + string(missionCfg.type);
    parts(end+1) = "solver_" + solverMode;
    parts(end+1) = "depOrb_" + string(tr.depOrbitIndex);
    parts(end+1) = "arrOrb_" + string(tr.arrOrbitIndex);
    parts(end+1) = "dt_" + local_num_str(tr.dt);
    parts(end+1) = "slots_" + string(slots_per_orbit);

    if isfield(tr,'depSlot') && ~isempty(tr.depSlot)
        parts(end+1) = "depSlot_" + string(tr.depSlot);
    end

    switch solverMode
        case "BALLISTIC"
            b = tr.ballistic;
            parts(end+1) = "tfg_"   + local_num_str(b.tf_guess);
            parts(end+1) = "dvmax_" + local_num_str(b.dv_max);
            parts(end+1) = "tflb_"  + local_num_str(b.tf_lb);
            parts(end+1) = "tfub_"  + local_num_str(b.tf_ub);
            if isfield(b,'dv_guess') && ~isempty(b.dv_guess)
                parts(end+1) = "dvg_" + local_vec_str(b.dv_guess);
            end

        case "TIME_OPT"
            p = tr.pmp;
            parts(end+1) = "m0_"    + local_num_str(p.m0);
            parts(end+1) = "Tmax_"  + local_num_str(p.Tmax);
            parts(end+1) = "ve_"    + local_num_str(p.ve);
            parts(end+1) = "tflb_"  + local_num_str(p.tf_lb);
            parts(end+1) = "tfub_"  + local_num_str(p.tf_ub);
            parts(end+1) = "tfg_"   + local_num_str(p.tf_guess);

        case "FUEL_OPT"
            % add fields here when implemented
    end

    rawKey = strjoin(parts, "__");
    rawKey = replace(rawKey, ".", "p");
    rawKey = replace(rawKey, "-", "m");
    rawKey = replace(rawKey, "+", "");
    rawKey = replace(rawKey, " ", "");
    rawKey = regexprep(rawKey, '[^a-zA-Z0-9_]', '_');

    if strlength(rawKey) > 180
        cacheKey = extractBefore(rawKey, 181);
    else
        cacheKey = rawKey;
    end
end

function s = local_num_str(x)
    s = string(sprintf('%.12g', x));
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

function create_sig_fig(fName, t, err, sig3, yLbl, w, h, cBnd, cErr, outDir, obsCount, maxObs)
    f = figure('Color','w','Units','inches','Position',[1 1 w h], ...
               'PaperUnits','inches','PaperPosition',[0 0 w h]);
    ax = axes(f);
    hold(ax,'on');
    box(ax,'on');
    set(ax,'TickLabelInterpreter','tex', 'Layer','top');

    % --- determine y-limits first ---
    yMax = max(abs([err(:); sig3(:)]));
    if ~isfinite(yMax) || yMax <= 0
        yMax = 1;
    end
    yPad = 0.08 * yMax;
    yLims = [-yMax-yPad, yMax+yPad];

    % --- grayscale background showing available observers ---
    if ~isempty(obsCount) && ~all(isnan(obsCount))
        obsCount = obsCount(:).';
        bg = repmat(obsCount, 2, 1);   % 2 rows so it fills the axes vertically

        imagesc(ax, t(:).', yLims, bg);
        set(ax, 'YDir', 'normal');

        colormap(ax, gray(max(2, maxObs+1)));
        clim(ax, [0 maxObs]);
        
        % make the background subtle
        bgHandle = findobj(ax, 'Type', 'Image');
        if ~isempty(bgHandle)
            bgHandle.AlphaData = 0.18;
        end
    end

    % --- main curves on top ---
    hB = plot(ax, t,  sig3, '-', 'Color', cBnd);
         plot(ax, t, -sig3, '-', 'Color', cBnd);
    hE = plot(ax, t,  err,  '-', 'Color', cErr);

    xlabel(ax, 't (TU)');
    ylabel(ax, yLbl);
    xlim(ax, [t(1) t(end)]);
    ylim(ax, yLims);

    % --- legend --- 
    lgd = legend(ax, [hE, hB], {'EKF error', '\pm 3\sigma bound'}, ...
        'Location', 'northeast');
    lgd.Box = 'on';
    lgd.ItemTokenSize = [18 12];

    % -- colorbar legend --- 
    cb = colorbar(ax);
    cb.Location = 'eastoutside';
    
    cb.Label.String = 'Available observers';
    cb.Ticks = 0:maxObs;
    cb.TickDirection = 'out';
    
    % ---- lock axes size ----
    ax.Units = 'normalized';
    ax.Position = [0.12 0.14 0.68 0.80];   % [left bottom width height]
    
    % ---- move colorbar further right ----
    cb.Units = 'normalized';
    cb.Position = [0.84 0.14 0.03 0.80];
    pos = cb.Position;
    pos(1) = pos(1) + 0.01;
    cb.Position = pos;

    % --- save figures ---
    exportgraphics(f, fullfile(outDir, fName), 'ContentType','image');
    savefig(f, fullfile(outDir, replace(fName, '.pdf', '.fig')));
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