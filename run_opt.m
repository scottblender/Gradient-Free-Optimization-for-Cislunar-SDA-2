% ---- run_optimization.m ---- %
clear; close all; clc;

% ---- PowerShell captures stdout to console.log ----
fprintf('RUN START: %s\n', string(datetime('now')));
drawnow;  % flush output

% load in filtered and sorted JPL data
S = load('JPL_CR3BP_OrbitCatalog.mat');
T1 = S.T;

% User-specified Inputs
% Options: 'GA', 'PSO', 'BAYESIAN', 'GAMULTIOBJ', 'DMOPSO', 'ABC', 'ACO'
OPTIMIZER_MODE = 'ACO'; % default

envMode = getenv("OPTIMIZER_MODE");
if ~isempty(envMode)
    OPTIMIZER_MODE = envMode;
end

OPTIMIZER_MODE = upper(string(OPTIMIZER_MODE));

% Number of observers to optimize
nvars = 3;

% ---------------- STOPPING CRITERIA ----------------
MAX_EVALS = 5000;

% Comparable stall window: 20 "stall iterations" for EACH algorithm
STALL_ITERS = 20;     % <-- you asked for 20
STALL_EPS   = 1e-2;   % abs improvement threshold
WARMUP_ITERS = 2;     % warmup (in iterations-worth); converted to evals per solver

% JPL Constants
mu = 1.215058560962404E-2;
LU = 384400;     % km
TU = 375695;     % seconds
VU = LU / TU;    % km/s

ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% --- EKF Parameters ---
pos_var = (1 / LU)^2;
vel_var = (10 / (VU * 1000))^2;
P_0_base = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);
Q_k = diag(repmat(1e-8, 6, 1));
R_k_base = diag([1e-8, 1e-8]);

% --- Lunar Gateway ICs ---
s_lg_ic = [1.02202108343387, 0, -0.182096487798513, 0, -0.103255420206012, 0]';
tspan_lg_ic = [0, 1.51110546287394];

% MILP-Implementation
num_orbits = height(T1); % number of candidate orbits
slots_per_orbit = 500;   % number of discrete slots per orbit

tf    = T1.("Period (TU) ");
states = T1.("state");
times  = T1.("time");
stabilities = T1.("Stability index  ");

orbit_database = cell(num_orbits, 1);

parfor i = 1:num_orbits
    t_raw = times{i};
    s_raw = states{i};
    period = tf(i);

    t_slots = linspace(0, period, slots_per_orbit)';

    [t_unique, idx_u] = unique(t_raw);
    s_unique = s_raw(idx_u, :);
    F = griddedInterpolant(t_unique, s_unique, 'spline');
    s_slots = F(t_slots);

    orbit_database{i} = s_slots;
end

% define EKF timestep
dt = 0.01; % TU ~ 1.04 hours
N_periods = 1;

% propagate truth trajectory (Lunar Gateway)
tspan_lg = tspan_lg_ic(1):dt:N_periods*tspan_lg_ic(2);
[t_lg, s_lg] = ode45(@(t,s) cr3bp_dynamics(t,s,mu), tspan_lg, s_lg_ic, ode_opts);

% set up data logging
dq = parallel.pool.DataQueue;
assignin('base', 'OptimizationLog', {});
afterEach(dq, @(data) append_log(data));

% --- HELPER FUNCTION --- %
function append_log(data)
    logCell = evalin('base', 'OptimizationLog');
    logCell{end+1,1} = data;
    assignin('base', 'OptimizationLog', logCell);
end

% set flag for single or multi-objective
opt_flag = 'SOO';
const_stabilities = parallel.pool.Constant(stabilities);
const_orbit_db = parallel.pool.Constant(orbit_database);

% ---- base objective ----
BaseObjFcn = @(x) objective_wrapper(x, const_orbit_db, const_stabilities, ...
                                   s_lg, t_lg, P_0_base, Q_k, R_k_base, mu, ...
                                   opt_flag, upper(OPTIMIZER_MODE), dq);

ObjFcnBuiltIn = BaseObjFcn;  % GA/PSO/BAYESIAN

RunTimer = tic;

switch upper(OPTIMIZER_MODE)

    % ---------------------------------------------------------------------
    case 'GA'
        fprintf('Starting Genetic Algorithm...\n');
        nVars = 6;

        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];
        IntCon = 1:nVars;

        pop = 60;
        maxGen = max(1, ceil(MAX_EVALS / pop));

        % 20 stall "iterations" == 20 generations-worth of evals
        STALL_M_SOLVER    = STALL_ITERS * pop;
        STALL_NMIN_SOLVER = max(50, WARMUP_ITERS * pop);

        stopper = Stopper(STALL_M_SOLVER, STALL_EPS, STALL_NMIN_SOLVER, MAX_EVALS, true);

        options = optimoptions('ga', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'PopulationSize', pop, ...
            'MaxGenerations', maxGen, ...
            'OutputFcn', @(options,state,flag) stopper.gaOutputFcn(options,state,flag));

        [x_best, min_cost] = ga(ObjFcnBuiltIn, nVars, [], [], [], [], LB, UB, [], IntCon, options);

    % ---------------------------------------------------------------------
    case 'PSO'
        fprintf('Starting Particle Swarm Optimization...\n');
        nVars = 6;

        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        swarm = 60;
        maxIter = max(1, ceil(MAX_EVALS / swarm));

        % 20 stall "iterations" == 20 PSO iterations-worth of evals
        STALL_M_SOLVER    = STALL_ITERS * swarm;
        STALL_NMIN_SOLVER = max(50, WARMUP_ITERS * swarm);

        stopper = Stopper(STALL_M_SOLVER, STALL_EPS, STALL_NMIN_SOLVER, MAX_EVALS, true);

        options = optimoptions('particleswarm', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'SwarmSize', swarm, ...
            'MaxIterations', maxIter, ...
            'OutputFcn', @(optimValues,state) stopper.psoOutputFcn(optimValues,state));

        [x_best, min_cost] = particleswarm(ObjFcnBuiltIn, nVars, LB, UB, options);
        x_best = round(x_best);

    % ---------------------------------------------------------------------
    case 'BAYESIAN'
        fprintf('Starting Bayesian Optimization...\n');

        vars = [];
        for i = 1:3
            vars = [vars, ...
                optimizableVariable(['Orbit',num2str(i)], [1, num_orbits], 'Type','integer'), ...
                optimizableVariable(['Slot',num2str(i)],  [1, slots_per_orbit], 'Type','integer')];
        end

        % Bayesopt doesn't have a natural "iteration" batch; treat 1 eval as 1 step.
        STALL_M_SOLVER    = STALL_ITERS;          % 20 evals of no improvement
        STALL_NMIN_SOLVER = max(50, WARMUP_ITERS);

        stopper = Stopper(STALL_M_SOLVER, STALL_EPS, STALL_NMIN_SOLVER, MAX_EVALS, true);

        results = bayesopt(ObjFcnBuiltIn, vars, ...
            'UseParallel', true, ...
            'IsObjectiveDeterministic', false, ...
            'MaxObjectiveEvaluations', MAX_EVALS, ...
            'OutputFcn', @(results,state) stopper.bayesOutputFcn(results,state));

        x_best = table2array(results.XAtMinObjective);
        min_cost = results.MinObjective;

    % ---------------------------------------------------------------------
    case 'ABC'
        fprintf('Starting Artificial Bee Colony Optimization...\n');

        nVars = 6;
        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        abc_opts.ColonySize      = 60;
        abc_opts.MaxIters        = max(1, ceil(MAX_EVALS / abc_opts.ColonySize));
        abc_opts.Limit           = 20;
        abc_opts.SlotsPerOrbit   = slots_per_orbit;
        abc_opts.UseParallel     = true;

        % ABC evals per iteration ≈ employed (nFood) + onlooker (nFood) = 2*nFood
        nFood = abc_opts.ColonySize/2;
        evalsPerIterABC = 2 * nFood;

        STALL_M_SOLVER    = STALL_ITERS * evalsPerIterABC;
        STALL_NMIN_SOLVER = max(50, WARMUP_ITERS * evalsPerIterABC);

        stopper = Stopper(STALL_M_SOLVER, STALL_EPS, STALL_NMIN_SOLVER, MAX_EVALS, true);

        [x_best, min_cost] = abc_discrete(ObjFcnBuiltIn, LB, UB, abc_opts, stopper);

    % ---------------------------------------------------------------------
    case 'ACO'
        fprintf('Starting Ant Colony Optimization...\n');

        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        aco_opts.nAnts    = 30;
        aco_opts.MaxIters = max(1, ceil(MAX_EVALS / aco_opts.nAnts));
        aco_opts.alpha    = 1.0;
        aco_opts.beta     = 1.0;
        aco_opts.rho      = 0.2;
        aco_opts.Q        = 1.0;
        aco_opts.UseParallel = true;

        % ACO evals per iteration = nAnts
        STALL_M_SOLVER    = STALL_ITERS * aco_opts.nAnts;
        STALL_NMIN_SOLVER = max(50, WARMUP_ITERS * aco_opts.nAnts);

        stopper = Stopper(STALL_M_SOLVER, STALL_EPS, STALL_NMIN_SOLVER, MAX_EVALS, true);

        [x_best, min_cost] = aco_discrete(ObjFcnBuiltIn, LB, UB, aco_opts, stopper);

    otherwise
        error("Unknown OPTIMIZER_MODE: %s", OPTIMIZER_MODE);
end

TotalRuntime = toc(RunTimer);
fprintf('Total Runtime: %.2f seconds\n', TotalRuntime);

fprintf("Objective evals used: %d\n", stopper.evalCount);
if stopper.stopFlag
    fprintf("Stopped early: %s\n", stopper.stopReason);
end

if strcmp(opt_flag, 'SOO')
    fprintf('\n--- FINAL RESULTS (%s) ---\n', OPTIMIZER_MODE);
    fprintf('Orbits: %s\n', mat2str(x_best(1:2:end)));
    fprintf('Slots:  %s\n', mat2str(x_best(2:2:end)));
    fprintf('Cost:   %.4f\n', min_cost);
end

fprintf('RUN END: %s\n', string(datetime('now')));
drawnow;

% -------------------------------------------------------------------------
% PGCP / PARALLEL CLEANUP (prevents MVM teardown asserts after -batch)
% -------------------------------------------------------------------------
try
    p = gcp('nocreate');   % <-- pool get current pool
    if ~isempty(p)
        delete(p);
    end
catch
end

drawnow;
pause(0.2);
