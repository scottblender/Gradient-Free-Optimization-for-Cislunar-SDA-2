% ---- run_optimization.m ---- %
clear; close all; clc;

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

% Stopping Criteria
MAX_EVALS = 500; 

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
slots_per_orbit = 500;          % number of discrete slots per orbit

tf    = T1.("Period (TU) ");
states = T1.("state");
times  = T1.("time");
stabilities = T1.("Stability index  ");


orbit_database = cell(num_orbits, 1);

parfor i = 1:num_orbits
    t_raw = times{i};
    s_raw = states{i};
    period = tf(i);
    % create vector of times
    t_slots = linspace(0, period, slots_per_orbit)';

    % interpolate propagated trajectories for specific time slots
    [t_unique, idx_u] = unique(t_raw);
    s_unique = s_raw(idx_u, :);
    F = griddedInterpolant(t_unique, s_unique, 'spline');
    s_slots = F(t_slots);
    % store the interpolated states in the orbit database
    orbit_database{i} = s_slots;
end

% define EKF timestep
dt = 0.01; % TU ~ 1.04 hours
N_periods = 1; % number of periods to propagate

% propagate truth trajectory (Lunar Gateway)
tspan_lg = tspan_lg_ic(1):dt:N_periods*tspan_lg_ic(2);
[t_lg, s_lg] = ode45(@(t,s) cr3bp_dynamics(t,s,mu), tspan_lg, s_lg_ic, ode_opts);

% set up data logging
% Create DataQueue on the Client
dq = parallel.pool.DataQueue;

% Initialize the Log Variable in Base Workspace
assignin('base', 'OptimizationLog', []); 

% Define Listener Callback (Runs on Client)
afterEach(dq, @(data) append_log(data));

% Helper function to update the variable in the Command Window Workspace
function append_log(data)
    logCell = evalin('base', 'OptimizationLog');
    logCell{end+1,1} = data; 
    assignin('base', 'OptimizationLog', logCell);
end

% set flag for single or multi-objective
opt_flag = 'SOO'; 
const_stabilities = parallel.pool.Constant(stabilities);
const_orbit_db = parallel.pool.Constant(orbit_database);
ObjFcn = @(x) objective_wrapper(x, const_orbit_db, const_stabilities, ...
                                          s_lg, t_lg, P_0_base, Q_k, R_k_base, mu, opt_flag, upper(OPTIMIZER_MODE), dq);
RunTimer = tic;
% swtich between optimizers
switch upper(OPTIMIZER_MODE)
    
    % ---------------------------------------------------------------------
    case 'GA'
        fprintf('Starting Genetic Algorithm...\n');
        nVars = 6;
        
        % Integers: [Orb1, Slt1, Orb2, Slt2, Orb3, Slt3]
        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];
        IntCon = 1:nVars; % All variables are integers

        pop = 60;  % keep consistent across runs
        maxGen = max(1, ceil(MAX_EVALS / pop));
        
        options = optimoptions('ga', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'PopulationSize', pop, ...
            'MaxGenerations', maxGen);

        [x_best, min_cost] = ga(ObjFcn, nVars, [], [], [], [], LB, UB, [], IntCon, options);

    % ---------------------------------------------------------------------
    case 'PSO'
        fprintf('Starting Particle Swarm Optimization...\n');
        nVars = 6;
        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];
        
        swarm = 60;
        maxIter = max(1, ceil(MAX_EVALS / swarm));
        
        options = optimoptions('particleswarm', ...
            'UseParallel', true, ...
            'Display', 'iter', ...
            'SwarmSize', swarm, ...
            'MaxIterations', maxIter);
        
        [x_best, min_cost] = particleswarm(ObjFcn, nVars, LB, UB, options);
        x_best = round(x_best);

    % ---------------------------------------------------------------------
    case 'BAYESIAN'
        fprintf('Starting Bayesian Optimization...\n');
        
        % Define Variables with Names
        vars = [];
        for i = 1:3 % For 3 observers
            vars = [vars, ...
                optimizableVariable(['Orbit',num2str(i)], [1, num_orbits], 'Type','integer'), ...
                optimizableVariable(['Slot',num2str(i)], [1, slots_per_orbit], 'Type','integer')];
        end
        
        results = bayesopt(ObjFcn, vars, ...
            'UseParallel', true, ...
            'IsObjectiveDeterministic', false, ...
            'MaxObjectiveEvaluations', MAX_EVALS);
                       
        x_best = table2array(results.XAtMinObjective);
        min_cost = results.MinObjective;

    % --------------------------------------------------------------------- 
    case 'GAMULTIOBJ'
        fprintf('Starting Multi-Objective Genetic Algorithm (NSGA-II)...\n');
        
        % Setup Bounds & Integers
        nVars = 6;
        LB = double([1, 1, 1, 1, 1, 1]);
        UB = double([num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit]);
        IntCon = 1:nVars; 

        % Options
        %    'ParetoFraction': How many individuals to keep on the front (0.3 = 30%)
        options = optimoptions('gamultiobj', ...
            'PopulationSize', 60, ...
            'ParetoFraction', 0.5, ... 
            'UseParallel', true, ...
            'Display', 'iter', ...
            'PlotFcn', @gaplotpareto); % Built-in 2D/3D Plotter

        % Run Optimizer
        [x_best, fval] = gamultiobj(ObjFcn, nVars, [], [], [], [], LB, UB, [], IntCon, options);
    
    % ---------------------------------------------------------------------  
    case 'DMOPSO'
        nVars = 6;
        LB = double([1, 1, 1, 1, 1, 1]);
        UB = double([num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit]);
        fprintf('Starting Custom Multi-Objective PSO...\n');
        [archive_X, archive_F] = dmopso(ObjFcn, nVars, LB, UB, 60, 100, N_STALL);
        
        % find best solution
        fval = archive_F;
        x_best = archive_X;
      
    % ---------------------------------------------------------------------
    case 'ABC'
        fprintf('Starting Artificial Bee Colony Optimization...\n');
        nVars = 6;

        % Integers: [Orb1, Slt1, Orb2, Slt2, Orb3, Slt3]
        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        % --- ABC options --- %
        abc_opts.ColonySize      = 60;
        abc_opts.MaxIters   = max(1, ceil(MAX_EVALS / abc_opts.ColonySize));
        abc_opts.Limit           = 20;   % scout trigger
        abc_opts.StallIters      = inf;
        abc_opts.SlotsPerOrbit   = slots_per_orbit;
        abc_opts.UseParallelInit = true; % only parallelize initial evaluation

        [x_best, min_cost] = abc_discrete(ObjFcn, LB, UB, abc_opts);

    % ---------------------------------------------------------------------
    case 'ACO'
        fprintf('Starting Ant Colony Optimization...\n');

        LB = [1, 1, 1, 1, 1, 1];
        UB = [num_orbits, slots_per_orbit, num_orbits, slots_per_orbit, num_orbits, slots_per_orbit];

        % --- ACO options --- %
        aco_opts.nAnts      = 30;
        aco_opts.MaxIters = max(1, ceil(MAX_EVALS / aco_opts.nAnts));
        aco_opts.alpha      = 1.0;   % pheromone influence
        aco_opts.beta       = 1.0;   % heuristic influence (keep 1 if no heuristic)
        aco_opts.rho        = 0.2;   % evaporation rate
        aco_opts.Q          = 1.0;   % deposit scale
        aco_opts.StallIters = inf;

        [x_best, min_cost] = aco_discrete(ObjFcn, LB, UB, aco_opts);

end

% runtime
TotalRuntime = toc(RunTimer);
fprintf('Total Runtime: %.2f seconds\n', TotalRuntime);

if strcmp(opt_flag, 'SOO')
    fprintf('\n--- FINAL RESULTS (%s) ---\n', OPTIMIZER_MODE);
    fprintf('Orbits: %s\n', mat2str(x_best(1:2:end)));
    fprintf('Slots:  %s\n', mat2str(x_best(2:2:end)));
    fprintf('Cost:   %.4f\n', min_cost);
else
    f_min = min(fval);
    f_max = max(fval);
    f_norm = (fval - f_min) ./ (f_max - f_min);
    
    % Calculate distance to Utopia point (0,0,0) in normalized space
    dist_to_utopia = sqrt(sum(f_norm.^2, 2));
    
    % Find the index of the "Knee" solution
    [~, idx_knee] = min(dist_to_utopia);
    
    % Extract the Actual (Raw) Costs for that solution
    knee_costs = fval(idx_knee, :);
    knee_vars  = x_best(idx_knee, :);
    
    % Print Results
    fprintf('\n--- KNEE POINT (Balanced Solution) ---\n');
    fprintf('Selected Row: %d\n', idx_knee);
    fprintf('RMSE (Log):   %.4f\n', knee_costs(1));
    fprintf('Det (Log):  %.4f\n', knee_costs(2));
    fprintf('Stability:    %.4f\n', knee_costs(3));
    fprintf('Orbits:       %s\n', mat2str(knee_vars(1:2:end)));
    fprintf('Slots:        %s\n', mat2str(knee_vars(2:2:end)));
end

% ==================== SAVE ARTIFACTS ====================
ts = char(datetime("now","Format","yyyy-MM-dd HH:mm:ss.SSS"));
mode = char(OPTIMIZER_MODE);

% Pull OptimizationLog from base workspace
logCell = evalin('base', 'OptimizationLog');

% ---- results struct ----
results = struct();
results.optimizer   = mode;
results.timestamp   = ts;
results.runtime_sec = TotalRuntime;
results.opt_flag    = opt_flag;

if strcmpi(opt_flag,'SOO')
    results.x_best   = x_best;
    results.min_cost = min_cost;
else
    results.x_best = x_best;
    results.fval   = fval;
end

% ---- Save MAT (everything) ----
ts = char(datetime("now","Format","yyyyMMdd_HHmmss"));
mode = char(OPTIMIZER_MODE);
matName = sprintf('results_%s_%s.mat', mode, ts);

save(matName, 'results', 'logCell', '-v7.3');

% ---- Save summary.txt (clean final) ----
fid = fopen('summary.txt',"w");
fprintf(fid, "Optimizer: %s\n", mode);
fprintf(fid, "Timestamp: %s\n", ts);
fprintf(fid, "Runtime (sec): %.3f\n", TotalRuntime);
fprintf(fid, "opt_flag: %s\n\n", opt_flag);

if strcmpi(opt_flag,'SOO')
    fprintf(fid, "x_best: %s\n", mat2str(x_best));
    fprintf(fid, "Orbits: %s\n", mat2str(x_best(1:2:end)));
    fprintf(fid, "Slots : %s\n", mat2str(x_best(2:2:end)));
    fprintf(fid, "min_cost: %.12f\n", min_cost);
else
    fprintf(fid, "Multi-objective run.\n");
end
fclose(fid);

% ---- Save OptimizationLog.csv ----
csvName = 'OptimizationLog.csv';

if isempty(logCell)
    fid = fopen(csvName,"w");
    fprintf(fid,"No OptimizationLog entries were recorded.\n");
    fclose(fid);
else
    try
        Tlog = struct2table([logCell{:}]);
        writetable(Tlog, csvName);
    catch ME
        warning("Failed to write CSV: %s", ME.message);
        fid = fopen(csvName,"w");
        fprintf(fid,"Failed to convert log to table. Dumping entries.\n\n");
        for i = 1:numel(logCell)
            fprintf(fid,"%s\n", evalc("disp(logCell{i})"));
        end
        fclose(fid);
    end
end

fprintf("\nSaved artifacts: %s, summary.txt, %s\n", matName, csvName);
% =========================================================
