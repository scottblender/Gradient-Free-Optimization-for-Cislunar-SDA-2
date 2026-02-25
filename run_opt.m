% ---- run_optimization.m ---- %
clear; close all; clc;

fprintf('RUN START: %s\n', string(datetime('now')));
drawnow;

% load in filtered and sorted JPL data
S  = load('JPL_CR3BP_OrbitCatalog.mat');
T1 = S.T;
t_lg = S.t_lg;
s_lg = S.s_lg;

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
slots_per_orbit = 50;        % number of discrete slots per orbit

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

% choose whether or not to include occlusion/exclusion
useScreening = true;

% struct to include or exclude cost components
costFlags = struct('J1', true, 'J2', true, 'J3', true);  % default - all true

ObjFcn = @(x) objective_wrapper(x, orbit_database, stabilities, s_lg, t_lg, P_0_base, Q_k, R_k_base, mu, LU, ...
    sunFcn, sun_min, moon_min, opt_flag, OPTIMIZER_MODE, dq, useScreening, costFlags);

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
    
        abc_opts.ColonySize      = 60;
        abc_opts.MaxIters        = MAX_ITERS;
        abc_opts.Limit           = 20;
        abc_opts.StallIters      = inf;        % or a number like 25
        abc_opts.SlotsPerOrbit   = slots_per_orbit;
        abc_opts.UseParallel     = true;       % <- now used for ALL phases
        abc_opts.UseParallelInit = true;
    
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
    
        % optional (recommended) extras supported by the updated script:
        aco_opts.TauMin              = 1e-12;
        aco_opts.UseIterBestDeposit  = true;
        aco_opts.IterBestWeight      = 1.0;   % try 0.5–2.0
        aco_opts.StallIters          = inf;   % or e.g., 25
    
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
    x_plot = x_best; % used for plotting
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
    x_plot = knee_vars; % used for plotting
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

% ---------------- Recompile results to plot  ----------------
x_plot = round(x_plot);

orbit_indices = x_plot(1:2:end);
slot_indices  = x_plot(2:2:end);
num_obs = numel(orbit_indices);

% quick safety clamps
for k = 1:num_obs
    orbit_indices(k) = max(1, min(orbit_indices(k), numel(orbit_database)));
    slot_indices(k)  = max(1, min(slot_indices(k), size(orbit_database{orbit_indices(k)},1)));
end

% ---------- build observer ICs from database (selected orbit/slot) ----------
observer_ICs = zeros(num_obs,6);
for k = 1:num_obs
    observer_ICs(k,:) = orbit_database{orbit_indices(k)}(slot_indices(k),:);
end

% ---------- 3D figure: plot each orbit (full period) from database ----------
figW = 3.5;     % inches (single-column)
figH = 3.0;     % inches
fig = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
             'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

ax = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% tighten the axes box inside the figure to reduce whitespace
ax.Units = 'normalized';
ax.Position = [0.12 0.12 0.84 0.84];   % [left bottom width height]

% ---------- run EKF to print out final screening count ------ %
[s_ekf, cov, screeningCount_final] = cr3bp_ekf(observer_ICs, s_lg, t_lg, ...
    P_0_base, Q_k, R_k_base, mu, LU, sunFcn, sun_min, moon_min, useScreening);

fprintf('\nFinal EKF screeningCount = %d\n', screeningCount_final);

% truth + EKF first
hEKF   = plot3(ax, s_ekf(:,1), s_ekf(:,2), s_ekf(:,3), '--', 'LineWidth', 2.0);

% observers from FULL catalog
cmap  = lines(max(1,num_obs));
hObs  = gobjects(num_obs,1);
hSlot = gobjects(num_obs,1);

for k = 1:num_obs
    iOrb   = orbit_indices(k);
    t_raw  = times{iOrb}(:);       % raw time samples (length Nr)
    s_raw  = states{iOrb};         % raw states (Nr x 6)
    Tper   = tf(iOrb);             % period (TU)

    % plot the raw orbit shape (full period as given)
    hObs(k) = plot3(ax, s_raw(:,1), s_raw(:,2), s_raw(:,3), ...
        'LineWidth', 1.4, 'Color', cmap(k,:));

    % mark selected slot on the RAW orbit by mapping slot -> phase time
    % (slot index is defined over slots_per_orbit)
    t_phase = (slot_indices(k)-1) / (slots_per_orbit-1) * Tper;

    % find closest raw time index (works even if t_raw isn't uniform)
    [~, j] = min(abs(t_raw - t_phase));

    hSlot(k) = plot3(ax, s_raw(j,1), s_raw(j,2), s_raw(j,3), ...
        'o', 'MarkerSize', 6, 'MarkerFaceColor', cmap(k,:), 'MarkerEdgeColor','k');
end

% primaries
rM = [1-mu, 0, 0];
hM = plot3(ax, rM(1), rM(2), rM(3), 'ko', 'MarkerSize',7,'MarkerFaceColor','#B0B0B0');
text(rM(1), rM(2), rM(3), '  Moon',  'Interpreter','latex');

% formatting
axis(ax,'equal'); view(ax,3);
xlabel(ax,'$x$ (LU)','Interpreter','latex');
ylabel(ax,'$y$ (LU)','Interpreter','latex');
zlabel(ax,'$z$ (LU)','Interpreter','latex');
set(ax,'FontName','Times New Roman','FontSize',15, ...
    'TickLabelInterpreter','latex','LineWidth',1.0);

ax.GridAlpha = 0.15;
ax.MinorGridAlpha = 0.10;
ax.XMinorGrid = 'on'; ax.YMinorGrid = 'on'; ax.ZMinorGrid = 'on';
axis equal

% legend
lgd = legend(ax, [hEKF, hObs(1), hSlot(1), hM], ...
    {'EKF estimate','Observer orbit','Selected slot','Moon'}, ...
    'Interpreter','latex', ...
    'Location','northeast');
lgd.Box = 'on';
lgd.FontSize = 15;
lgd.ItemTokenSize = [16 12];

% export
exportgraphics(fig,'fig_traj3d_rawS.pdf','ContentType','image');
exportgraphics(fig,'fig_traj3d_rawS.png','Resolution',600);

% ---------- 3-sigma position and velocity bounds ----------
Nf = size(cov,1);
sig = zeros(Nf,6);
for k = 1:Nf
    Pk = squeeze(cov(k,:,:));
    sig(k,:) = sqrt(max(diag(Pk),0));  % guard numerical negatives
end
sig3 = 3*sig;

% time vector for plotting (TU)
t = t_lg(:);

% EKF error (truth - estimate) in LU and LU/TU
err = s_ekf(:,1:6) - s_lg(:,1:6);

% convert errors to km and km/s
err_pos_km  = err(:,1:3) * LU;
err_vel_kms = err(:,4:6) * VU;

% convert 3-sigma bounds to km and km/s
sig3_pos_km  = sig3(:,1:3) * LU;
sig3_vel_kms = sig3(:,4:6) * VU;

% position 3-sigma (x)
figSigX = figure('Color','w','Units','inches','Position',[1 1 3.5 2.6], ...
                 'PaperUnits','inches','PaperPosition',[0 0 3.5 2.6]);
axx = axes(figSigX); hold(axx,'on'); grid(axx,'on'); box(axx,'on');
plot(axx, t,  sig3_pos_km(:,1), 'r-', 'LineWidth',1.8);
plot(axx, t, -sig3_pos_km(:,1), 'r-', 'LineWidth',1.8);
plot(axx, t,  err_pos_km(:,1),  'LineWidth',1.6);
xlabel(axx,'$t$ (TU)','Interpreter','latex');
ylabel(axx,'$e_x$ (km)','Interpreter','latex');
set(axx,'FontName','Times New Roman','FontSize',15,'TickLabelInterpreter','latex','LineWidth',1.0);
exportgraphics(figSigX,'fig_3sig_x.pdf','ContentType','image');

% position 3-sigma (y)
figSigY = figure('Color','w','Units','inches','Position',[1 1 3.5 2.6], ...
                 'PaperUnits','inches','PaperPosition',[0 0 3.5 2.6]);
axy = axes(figSigY); hold(axy,'on'); grid(axy,'on'); box(axy,'on');
plot(axy, t,  sig3_pos_km(:,2), 'r-', 'LineWidth',1.8);
plot(axy, t, -sig3_pos_km(:,2), 'r-', 'LineWidth',1.8);
plot(axy, t,  err_pos_km(:,2),  'LineWidth',1.6);
xlabel(axy,'$t$ (TU)','Interpreter','latex');
ylabel(axy,'$e_y$ (km)','Interpreter','latex');
set(axy,'FontName','Times New Roman','FontSize',15,'TickLabelInterpreter','latex','LineWidth',1.0);
exportgraphics(figSigY,'fig_3sig_y.pdf','ContentType','image');

% position 3-sigma (z)
figSigZ = figure('Color','w','Units','inches','Position',[1 1 3.5 2.6], ...
                 'PaperUnits','inches','PaperPosition',[0 0 3.5 2.6]);
axz = axes(figSigZ); hold(axz,'on'); grid(axz,'on'); box(axz,'on');
plot(axz, t,  sig3_pos_km(:,3), 'r-', 'LineWidth',1.8);
plot(axz, t, -sig3_pos_km(:,3), 'r-', 'LineWidth',1.8);
plot(axz, t,  err_pos_km(:,3),  'LineWidth',1.6);
xlabel(axz,'$t$ (TU)','Interpreter','latex');
ylabel(axz,'$e_z$ (km)','Interpreter','latex');
set(axz,'FontName','Times New Roman','FontSize',15,'TickLabelInterpreter','latex','LineWidth',1.0);
exportgraphics(figSigZ,'fig_3sig_z.pdf','ContentType','image');

% velocity 3-sigma (vx)
figSigVx = figure('Color','w','Units','inches','Position',[1 1 3.5 2.6], ...
                  'PaperUnits','inches','PaperPosition',[0 0 3.5 2.6]);
axvx = axes(figSigVx); hold(axvx,'on'); grid(axvx,'on'); box(axvx,'on');
plot(axvx, t,  sig3_vel_kms(:,1), 'r-', 'LineWidth',1.8);
plot(axvx, t, -sig3_vel_kms(:,1), 'r-', 'LineWidth',1.8);
plot(axvx, t,  err_vel_kms(:,1),  'LineWidth',1.6);
xlabel(axvx,'$t$ (TU)','Interpreter','latex');
ylabel(axvx,'$e_{v_x}$ (km/s)','Interpreter','latex');
set(axvx,'FontName','Times New Roman','FontSize',15,'TickLabelInterpreter','latex','LineWidth',1.0);
exportgraphics(figSigVx,'fig_3sig_vx.pdf','ContentType','image');

% velocity 3-sigma (vy)
figSigVy = figure('Color','w','Units','inches','Position',[1 1 3.5 2.6], ...
                  'PaperUnits','inches','PaperPosition',[0 0 3.5 2.6]);
axvy = axes(figSigVy); hold(axvy,'on'); grid(axvy,'on'); box(axvy,'on');
plot(axvy, t,  sig3_vel_kms(:,2), 'r-', 'LineWidth',1.8);
plot(axvy, t, -sig3_vel_kms(:,2), 'r-', 'LineWidth',1.8);
plot(axvy, t,  err_vel_kms(:,2),  'LineWidth',1.6);
xlabel(axvy,'$t$ (TU)','Interpreter','latex');
ylabel(axvy,'$e_{v_y}$ (km/s)','Interpreter','latex');
set(axvy,'FontName','Times New Roman','FontSize',15,'TickLabelInterpreter','latex','LineWidth',1.0);
exportgraphics(figSigVy,'fig_3sig_vy.pdf','ContentType','image');

% velocity 3-sigma (vz)
figSigVz = figure('Color','w','Units','inches','Position',[1 1 3.5 2.6], ...
                  'PaperUnits','inches','PaperPosition',[0 0 3.5 2.6]);
axvz = axes(figSigVz); hold(axvz,'on'); grid(axvz,'on'); box(axvz,'on');
plot(axvz, t,  sig3_vel_kms(:,3), 'r-', 'LineWidth',1.8);
plot(axvz, t, -sig3_vel_kms(:,3), 'r-', 'LineWidth',1.8);
plot(axvz, t,  err_vel_kms(:,3),  'LineWidth',1.6);
xlabel(axvz,'$t$ (TU)','Interpreter','latex');
ylabel(axvz,'$e_{v_z}$ (km/s)','Interpreter','latex');
set(axvz,'FontName','Times New Roman','FontSize',15,'TickLabelInterpreter','latex','LineWidth',1.0);
exportgraphics(figSigVz,'fig_3sig_vz.pdf','ContentType','image');

% ---------- RMSE + covariance summaries ----------
rmse_pos = sqrt(mean(sum((s_ekf(:,1:3) - s_lg(:,1:3)).^2,2)));
rmse_vel = sqrt(mean(sum((s_ekf(:,4:6) - s_lg(:,4:6)).^2,2)));

% convert RMSE to km and km/s
rmse_pos_km  = rmse_pos * LU;
rmse_vel_kms = rmse_vel * VU;

% position covariance metrics
detPpos = zeros(Nf,1);
for k = 1:Nf
    Pk = squeeze(cov(k,:,:));
    Ppos = Pk(1:3,1:3);
    detPpos(k) = det(Ppos);
end

% convert det(P_pos) from LU^6 to km^6
detPpos_km6 = detPpos * (LU^6);

fprintf('\n--- EKF PERFORMANCE ---\n');
fprintf('RMSE position (km):     %.6e\n', rmse_pos_km);
fprintf('RMSE velocity (km/s):   %.6e\n', rmse_vel_kms);
fprintf('Mean det(P_pos) (km^6): %.6e\n', mean(detPpos_km6));