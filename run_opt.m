% ---- run_optimization.m ---- %
clear; close all; clc;

fprintf('RUN START: %s\n', string(datetime('now')));
drawnow;

% load in filtered and sorted JPL data
S  = load('JPL_CR3BP_OrbitCatalog.mat');
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

% Stopping Criteria (max iterations for all except Bayesian)
MAX_ITERS = 100;
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
slots_per_orbit = 100;        % number of discrete slots per orbit

tf          = T1.("Period (TU) ");
states      = T1.("state");
times       = T1.("time");
stabilities = T1.("Stability index  ");

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

% define EKF timestep
dt        = 0.01; % TU ~ 1.04 hours
N_periods = 1;

% propagate truth trajectory (Lunar Gateway)
tspan_lg     = tspan_lg_ic(1):dt:N_periods*tspan_lg_ic(2);
[t_lg, s_lg] = ode45(@(t,s) cr3bp_dynamics(t,s,mu), tspan_lg, s_lg_ic, ode_opts);

% set up data logging (in-memory only)
dq = parallel.pool.DataQueue;
assignin('base', 'OptimizationLog', {});
afterEach(dq, @(data) append_log(data));

% --- helper function --- %
function append_log(data)
    logCell = evalin('base', 'OptimizationLog');
    logCell{end+1,1} = data;
    assignin('base', 'OptimizationLog', logCell);
end

% set flag for single or multi-objective
opt_flag          = 'SOO';
const_stabilities = parallel.pool.Constant(stabilities);
const_orbit_db    = parallel.pool.Constant(orbit_database);

% ---- Visibility Parameters ----
sun_min_deg  = 20;   % Sun exclusion angle (deg)
moon_min_deg = 10;   % Moon exclusion angle (deg)

sun_min  = deg2rad(sun_min_deg);
moon_min = deg2rad(moon_min_deg);

theta0 = 0;            % initial phase angle (rad)
i_sun  = deg2rad(0);    % keep planar for now

sunFcn = @(t) sun_pos_bc4bp(t, LU, TU, theta0, i_sun);

ObjFcn = @(x) objective_wrapper(x, const_orbit_db, const_stabilities, ...
                               s_lg, t_lg, P_0_base, Q_k, R_k_base, ...
                               mu, LU, sunFcn, sun_min, moon_min, ...
                               opt_flag, upper(OPTIMIZER_MODE), dq);
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

        options = optimoptions('ga', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'PopulationSize', pop, ...
            'MaxGenerations', MAX_ITERS);

        [x_best, min_cost] = ga(ObjFcn, nVars, [], [], [], [], LB, UB, [], IntCon, options);

    % ---------------------------------------------------------------------
    case 'PSO'
        fprintf('Starting Particle Swarm Optimization...\n');
        nVars = 6;

        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        swarm = 60;

        options = optimoptions('particleswarm', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'SwarmSize', swarm, ...
            'MaxIterations', MAX_ITERS);

        [x_best, min_cost] = particleswarm(ObjFcn, nVars, LB, UB, options);
        x_best = round(x_best);

    % ---------------------------------------------------------------------
    case 'BAYESIAN'
        fprintf('Starting Bayesian Optimization...\n');

        vars = [];
        for i = 1:3
            vars = [vars, ...
                optimizableVariable(['Orbit',num2str(i)], [1, num_orbits], 'Type','integer'), ...
                optimizableVariable(['Slot', num2str(i)], [1, slots_per_orbit], 'Type','integer')];
        end

        results = bayesopt(ObjFcn, vars, ...
            'UseParallel', true, ...
            'IsObjectiveDeterministic', false, ...
            'MaxObjectiveEvaluations', MAX_EVALS);

        x_best   = table2array(results.XAtMinObjective);
        min_cost = results.MinObjective;

    % ---------------------------------------------------------------------
    case 'GAMULTIOBJ'
        fprintf('Starting Multi-Objective Genetic Algorithm (NSGA-II)...\n');

        nVars = 6;
        LB = double([1, 1, 1, 1, 1, 1]);
        UB = double([num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit]);
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

    % ---------------------------------------------------------------------
    case 'DMOPSO'
        fprintf('Starting Custom Multi-Objective PSO...\n');

        nVars = 6;
        LB = double([1, 1, 1, 1, 1, 1]);
        UB = double([num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit]);

        swarmSize  = 60;
        maxIter    = MAX_ITERS;
        stallIters = inf;

        [archive_X, archive_F] = dmopso(ObjFcn, nVars, LB, UB, swarmSize, maxIter, stallIters);
        fval   = archive_F;
        x_best = archive_X;

    % ---------------------------------------------------------------------
    case 'ABC'
        fprintf('Starting Artificial Bee Colony Optimization...\n');

        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        abc_opts.ColonySize  = 60;
        abc_opts.MaxIters    = MAX_ITERS;
        abc_opts.Limit       = 20;
        abc_opts.UseParallel = true;

        [x_best, min_cost] = abc_discrete(ObjFcn, LB, UB, abc_opts);

    % ---------------------------------------------------------------------
    case 'ACO'
        fprintf('Starting Ant Colony Optimization...\n');

        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        aco_opts.nAnts       = 60;
        aco_opts.MaxIters    = MAX_ITERS;
        aco_opts.alpha       = 1.0;
        aco_opts.beta        = 1.0;
        aco_opts.rho         = 0.2;
        aco_opts.Q           = 1.0;
        aco_opts.UseParallel = true;

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
end

fprintf('RUN END: %s\n', string(datetime('now')));
drawnow;

% --- parallel pool cleanup --- %
% Explicitly delete the pool at the end of a -batch run to prevent teardown asserts.
try
    p = gcp('nocreate');   % get current parallel pool (if it exists)
    if ~isempty(p)
        delete(p);         % shut down pool cleanly
    end
catch
end

drawnow;
pause(0.2);
