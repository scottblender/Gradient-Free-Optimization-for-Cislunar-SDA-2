% ---- run_optimization.m ---- %
clear; close all; clc;

fprintf('RUN START: %s\n', string(datetime('now')));
drawnow;

% defaults for figures (16 pt everywhere, bold)
set(groot, ...
    'defaultAxesFontSize',16, ...
    'defaultAxesFontWeight','bold', ...
    'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontSize',16, ...
    'defaultTextFontWeight','bold', ...
    'defaultTextFontName','Times New Roman', ...
    'defaultLegendFontSize',16, ...
    'defaultLegendFontWeight','bold', ...
    'defaultAxesLabelFontSizeMultiplier',1.0, ...
    'defaultAxesTitleFontSizeMultiplier',1.0, ...
    'defaultLineLineWidth',1.8);

% load in filtered and sorted JPL data
S  = load('JPL_CR3BP_OrbitCatalog.mat');
T1 = S.T;
t_lg = S.t_lg;
s_lg = S.s_lg;

% User-specified Inputs
% Options: 'GA', 'PSO', 'BAYESIAN', 'GAMULTIOBJ', 'DMOPSO', 'ABC', 'ACO'
OPTIMIZER_MODE = 'ABC'; % default

envMode = getenv("OPTIMIZER_MODE");
if ~isempty(envMode)
    OPTIMIZER_MODE = envMode;
end
OPTIMIZER_MODE = upper(string(OPTIMIZER_MODE));

% Number of observers to optimize
nvars = 3;

% Stopping Criteria (max iterations for all except Bayesian)
MAX_ITERS = 10;
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

% artifacts folders + console capture
runDirEnv = getenv("RUN_DIR");
if ~isempty(runDirEnv)
    if ~exist(runDirEnv,'dir')
        mkdir(runDirEnv); % <-- MINIMAL FIX: ensure the abc_J1111 folder exists
    end
    cd(runDirEnv);        % <-- everything should stay inside this folder
end

RunDir = pwd;

% Base artifacts folder (stays under RUN_DIR)
ArtDir  = fullfile(RunDir, "artifacts");
if ~exist(ArtDir,'dir'),  mkdir(ArtDir);  end

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

% if using powershell script
v = getenv("MAX_ITERS"); if ~isempty(v), MAX_ITERS = str2double(v); end

v = getenv("USE_SCREENING");
if ~isempty(v), useScreening = (str2double(v) ~= 0); end

vj1 = getenv("USE_J1"); if ~isempty(vj1), costFlags.J1 = (str2double(vj1) ~= 0); end
vj2 = getenv("USE_J2"); if ~isempty(vj2), costFlags.J2 = (str2double(vj2) ~= 0); end
vj3 = getenv("USE_J3"); if ~isempty(vj3), costFlags.J3 = (str2double(vj3) ~= 0); end

% --- MINIMAL ADD: run tag + per-run folders + one excel file ---
vseed = getenv("SEED"); if isempty(vseed), vseed = "0"; end
seedVal = str2double(vseed); if isnan(seedVal), seedVal = 0; end

RUN_TAG = sprintf('%s_scr%d_J%d%d%d_seed%03d', char(OPTIMIZER_MODE), ...
    double(useScreening), double(costFlags.J1), double(costFlags.J2), double(costFlags.J3), seedVal);

EXCEL_FILE = fullfile(ArtDir, "ExperimentSummary.xlsx");

RunArtDir  = fullfile(ArtDir, RUN_TAG);          % <-- per-run root (under abc_J1111/artifacts/RUN_TAG)
FigDir     = fullfile(RunArtDir, "figs");
DataDir    = fullfile(RunArtDir, "data");        % <-- FIX: per-run data goes here
LogDir     = fullfile(RunArtDir, "logs");        % <-- FIX: per-run logs go here
if ~exist(RunArtDir,'dir'), mkdir(RunArtDir); end
if ~exist(FigDir,'dir'),    mkdir(FigDir);    end
if ~exist(DataDir,'dir'),   mkdir(DataDir);   end
if ~exist(LogDir,'dir'),    mkdir(LogDir);    end

% make diary filename unique so it doesn't overwrite (and keep it IN the per-run folder)
try
    diaryFile = fullfile(LogDir, "matlab_diary_" + string(datetime("now","Format","yyyyMMdd_HHmmss")) + ".txt");
    diary(diaryFile);
    diary on
catch
end
% ---------------------------------------------------------------

% set up data logging (in-memory only)
dq = parallel.pool.DataQueue;
assignin('base', 'OptimizationLog', {});
afterEach(dq, @(data) append_log(data));

% --- helper function ---
function append_log(data)
    logCell = evalin('base', 'OptimizationLog');
    logCell{end+1,1} = data;
    assignin('base', 'OptimizationLog', logCell);
end

% set flag for single or multi-objective
opt_flag          = 'SOO';
const_stabilities = parallel.pool.Constant(stabilities);
const_orbit_db    = parallel.pool.Constant(orbit_database);

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
            'MaxGenerations', MAX_ITERS, ...
            'MaxStallGenerations', Inf, ...
            'FunctionTolerance', 0, ...
            'ConstraintTolerance', 0, ...
            'FitnessLimit', -Inf);

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

[s_ekf, cov, screeningCount_final] = cr3bp_ekf(observer_ICs, s_lg, t_lg, ...
    P_0_base, Q_k, R_k_base, mu, LU, sunFcn, sun_min, moon_min, useScreening);

fprintf('\nFinal EKF screeningCount = %d\n', screeningCount_final);

% ---------------- observer metadata (family, etc.) ----------------
% (saved into the Excel file as an additional sheet)
familyColName = "";
vars = string(T1.Properties.VariableNames);
cand = ["Family","family","Orbit family","OrbitFamily","Family  ","FamilyName"];
for c = cand
    if any(vars == c)
        familyColName = c;
        break;
    end
end

obs_family = strings(num_obs,1);
if strlength(familyColName) > 0
    try
        famCol = T1.(familyColName);
        if iscell(famCol)
            obs_family = string(famCol(orbit_indices));
        else
            obs_family = string(famCol(orbit_indices));
        end
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
figH = 8;
fig = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
             'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

ax = axes(fig); hold(ax,'on');  box(ax,'on');
ax.Units = 'normalized';
ax.Position = [0.12 0.12 0.84 0.84];

% minimal: do not override font defaults; keep styling only
set(ax, 'TickLabelInterpreter','tex', 'LineWidth',1.0, 'Layer','top');

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
    hObs(k) = plot3(ax, s_raw(:,1), s_raw(:,2), s_raw(:,3), ...
        'LineWidth', 1.9, 'Color', cmap(k,:));
end

% Moon location
rM = [1-mu, 0, 0];
hM = plot3(ax, rM(1), rM(2), rM(3), 'ko', ...
    'MarkerSize',9, 'MarkerFaceColor',[0.70 0.70 0.70], 'LineWidth',1.0);

% L1/L2 points (collinear) + plot
[xL1, xL2] = cr3bp_L1L2(mu);
hL1 = plot3(ax, xL1, 0, 0, 'kd', 'MarkerSize',8, 'MarkerFaceColor',[0.85 0.85 0.85], 'LineWidth',1.0);
hL2 = plot3(ax, xL2, 0, 0, 'ks', 'MarkerSize',8, 'MarkerFaceColor',[0.85 0.85 0.85], 'LineWidth',1.0);

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

plot3(ax, s_lg_ic(1), s_lg_ic(2), s_lg_ic(3), 'o', ...
    'MarkerSize',8, 'MarkerFaceColor', 'red', 'MarkerEdgeColor','k');

xlabel(ax,'x (LU)');
ylabel(ax,'y (LU)');
zlabel(ax,'z (LU)');

axis(ax,'equal');
axis(ax,'tight');
ax.DataAspectRatio = [1 1 1];

% --- enforce equal tick spacing on x/y/z  ---
xl = xlim(ax); yl = ylim(ax); zl = zlim(ax);
xr = xl(2)-xl(1); yr = yl(2)-yl(1); zr = zl(2)-zl(1);
tickStep = max([xr, yr, zr]) / 8;          % slightly denser than before
tickStep = max(tickStep, eps);             % guard

% make tick step a "nice" decimal
pow10 = 10^floor(log10(tickStep));
nice = tickStep / pow10;
if     nice <= 1,   nice = 1;
elseif nice <= 2,   nice = 2;
elseif nice <= 2.5, nice = 2.5;
elseif nice <= 5,   nice = 5;
else                nice = 10;
end
tickStep = nice * pow10;

xTick0 = ceil(xl(1)/tickStep)*tickStep;
yTick0 = ceil(yl(1)/tickStep)*tickStep;
zTick0 = ceil(zl(1)/tickStep)*tickStep;

ax.XTick = xTick0:tickStep:xl(2);
ax.YTick = yTick0:tickStep:yl(2);
ax.ZTick = zTick0:tickStep:zl(2);

legHandles = [hEKF; hObs(:); hM; hL1; hL2];
legLabels = cell(num_obs + 4, 1);
legLabels{1} = 'EKF estimate';
for k = 1:num_obs
    legLabels{1+k} = sprintf('Observer %d orbit', k);
end
legLabels{end-2} = 'Moon';
legLabels{end-1} = 'L1';
legLabels{end}   = 'L2';

lgd = legend(ax, legHandles, legLabels, 'Location','northeast');
lgd.Box = 'on';
lgd.ItemTokenSize = [18 12];
if num_obs > 3
    lgd.NumColumns = 2;
end

exportgraphics(fig, fullfile(FigDir,'fig_traj3d.pdf'), 'ContentType','image');

% ---------------- 3-sigma plots ----------------
Nf = size(cov,1);
sig = zeros(Nf,6);
for k = 1:Nf
    Pk = squeeze(cov(k,:,:));
    sig(k,:) = sqrt(max(diag(Pk),0));
end
sig3 = 3*sig;

t = t_lg(:);
err = s_ekf(:,1:6) - s_lg(:,1:6);

err_pos_km  = err(:,1:3) * LU;
err_vel_kms = err(:,4:6) * VU;

sig3_pos_km  = sig3(:,1:3) * LU;
sig3_vel_kms = sig3(:,4:6) * VU;

cBound = [0.85 0.10 0.10];
cErr   = [0.00 0.45 0.74];

figSigX = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);
axx = axes(figSigX); hold(axx,'on'); box(axx,'on');
set(axx,'TickLabelInterpreter','tex','LineWidth',1.0,'Layer','top');
hBxP = plot(axx, t,  sig3_pos_km(:,1), '-', 'LineWidth',1.9, 'Color', cBound);
plot(axx, t, -sig3_pos_km(:,1), '-', 'LineWidth',1.9, 'Color', cBound);
hEx  = plot(axx, t,  err_pos_km(:,1),  '-',  'LineWidth',1.7, 'Color', cErr);
xlabel(axx,'t (TU)');
ylabel(axx,'e_x (km)');
xlim(axx,[t(1) t(end)]);
format_sig_axes(axx);
lgd = legend(axx, [hEx, hBxP], {'EKF error','\pm 3\sigma bound'}, 'Location','northeast');
lgd.Box = 'on'; lgd.ItemTokenSize = [18 12];
exportgraphics(figSigX, fullfile(FigDir,'fig_3sig_x.pdf'), 'ContentType','image');

figSigY = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);
axy = axes(figSigY); hold(axy,'on'); box(axy,'on');
set(axy,'TickLabelInterpreter','tex','LineWidth',1.0,'Layer','top');
hByP = plot(axy, t,  sig3_pos_km(:,2), '-', 'LineWidth',1.9, 'Color', cBound);
plot(axy, t, -sig3_pos_km(:,2), '-', 'LineWidth',1.9, 'Color', cBound);
hEy  = plot(axy, t,  err_pos_km(:,2),  '-',  'LineWidth',1.7, 'Color', cErr);
xlabel(axy,'t (TU)');
ylabel(axy,'e_y (km)');
xlim(axy,[t(1) t(end)]);
format_sig_axes(axy);
lgd = legend(axy, [hEy, hByP], {'EKF error','\pm 3\sigma bound'}, 'Location','northeast');
lgd.Box = 'on'; lgd.ItemTokenSize = [18 12];
exportgraphics(figSigY, fullfile(FigDir,'fig_3sig_y.pdf'), 'ContentType','image');

figSigZ = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);
axz = axes(figSigZ); hold(axz,'on'); box(axz,'on');
set(axz,'TickLabelInterpreter','tex','LineWidth',1.0,'Layer','top');
hBzP = plot(axz, t,  sig3_pos_km(:,3), '-', 'LineWidth',1.9, 'Color', cBound);
plot(axz, t, -sig3_pos_km(:,3), '-', 'LineWidth',1.9, 'Color', cBound);
hEz  = plot(axz, t,  err_pos_km(:,3),  '-',  'LineWidth',1.7, 'Color', cErr);
xlabel(axz,'t (TU)');
ylabel(axz,'e_z (km)');
xlim(axz,[t(1) t(end)]);
format_sig_axes(axz);
lgd = legend(axz, [hEz, hBzP], {'EKF error','\pm 3\sigma bound'}, 'Location','northeast');
lgd.Box = 'on'; lgd.ItemTokenSize = [18 12];
exportgraphics(figSigZ, fullfile(FigDir,'fig_3sig_z.pdf'), 'ContentType','image');

figSigVx = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);
axvx = axes(figSigVx); hold(axvx,'on'); box(axvx,'on');
set(axvx,'TickLabelInterpreter','tex','LineWidth',1.0,'Layer','top');
hBVxP = plot(axvx, t,  sig3_vel_kms(:,1), '-', 'LineWidth',1.9, 'Color', cBound);
plot(axvx, t, -sig3_vel_kms(:,1), '-', 'LineWidth',1.9, 'Color', cBound);
hEVx  = plot(axvx, t,  err_vel_kms(:,1),  '-',  'LineWidth',1.7, 'Color', cErr);
xlabel(axvx,'t (TU)');
ylabel(axvx,'e_{v_x} (km/s)');
xlim(axvx,[t(1) t(end)]);
format_sig_axes(axvx);
lgd = legend(axvx, [hEVx, hBVxP], {'EKF error','\pm 3\sigma bound'}, 'Location','northeast');
lgd.Box = 'on'; lgd.ItemTokenSize = [18 12];
exportgraphics(figSigVx, fullfile(FigDir,'fig_3sig_vx.pdf'), 'ContentType','image');

figSigVy = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);
axvy = axes(figSigVy); hold(axvy,'on');  box(axvy,'on');
set(axvy,'TickLabelInterpreter','tex','LineWidth',1.0,'Layer','top');
hBVyP = plot(axvy, t,  sig3_vel_kms(:,2), '-', 'LineWidth',1.9, 'Color', cBound);
plot(axvy, t, -sig3_vel_kms(:,2), '-', 'LineWidth',1.9, 'Color', cBound);
hEVy  = plot(axvy, t,  err_vel_kms(:,2),  '-',  'LineWidth',1.7, 'Color', cErr);
xlabel(axvy,'t (TU)');
ylabel(axvy,'e_{v_y} (km/s)');
xlim(axvy,[t(1) t(end)]);
format_sig_axes(axvy);
lgd = legend(axvy, [hEVy, hBVyP], {'EKF error','\pm 3\sigma bound'}, 'Location','northeast');
lgd.Box = 'on'; lgd.ItemTokenSize = [18 12];
exportgraphics(figSigVy, fullfile(FigDir,'fig_3sig_vy.pdf'), 'ContentType','image');

figSigVz = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);
axvz = axes(figSigVz); hold(axvz,'on');  box(axvz,'on');
set(axvz,'TickLabelInterpreter','tex','LineWidth',1.0,'Layer','top');
hBVzP = plot(axvz, t,  sig3_vel_kms(:,3), '-', 'LineWidth',1.9, 'Color', cBound);
plot(axvz, t, -sig3_vel_kms(:,3), '-', 'LineWidth',1.9, 'Color', cBound);
hEVz  = plot(axvz, t,  err_vel_kms(:,3),  '-',  'LineWidth',1.7, 'Color', cErr);
xlabel(axvz,'t (TU)');
ylabel(axvz,'e_{v_z} (km/s)');
xlim(axvz,[t(1) t(end)]);
format_sig_axes(axvz);
lgd = legend(axvz, [hEVz, hBVzP], {'EKF error','\pm 3\sigma bound'}, 'Location','northeast');
lgd.Box = 'on'; lgd.ItemTokenSize = [18 12];
exportgraphics(figSigVz, fullfile(FigDir,'fig_3sig_vz.pdf'), 'ContentType','image');

% ---------------- print statements ----------------
rmse_pos = sqrt(mean(sum((s_ekf(:,1:3) - s_lg(:,1:3)).^2,2)));
rmse_vel = sqrt(mean(sum((s_ekf(:,4:6) - s_lg(:,4:6)).^2,2)));

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

% --- one Excel file (Summary + per-run cost history sheet + per-run observer sheet) ---
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

    % cost history sheet name
    sheetName = matlab.lang.makeValidName(RUN_TAG);
    sheetName = replace(sheetName,"_","");
    if strlength(sheetName) > 31
        sheetName = extractBefore(sheetName, 32);
    end
    writetable(histTbl, EXCEL_FILE, 'Sheet', char(sheetName));

    % observer sheet name
    obsSheet = matlab.lang.makeValidName(RUN_TAG + "_obs");
    obsSheet = replace(obsSheet,"_","");
    if strlength(obsSheet) > 31
        obsSheet = extractBefore(obsSheet, 32);
    end
    writetable(obsTbl, EXCEL_FILE, 'Sheet', char(obsSheet));

catch ME
    fprintf(2,"WARNING: failed to write ExperimentSummary.xlsx: %s\n", ME.message);
end
% -------------------------------------------------------------------------

diary off

function format_sig_axes(axh)
% Keep layout + interpreter, do not override global font defaults
set(axh, 'TickLabelInterpreter','tex', 'LineWidth',1.0, 'Layer','top');
axh.Units = 'normalized';
axh.Position = [0.14 0.16 0.82 0.78];
end

function [xL1, xL2] = cr3bp_L1L2(mu)
% Solve for L1 and L2 x-locations (y=z=0) in standard CR3BP rotating frame
% Primaries at x1=-mu, x2=1-mu.
f = @(x) x ...
    - (1-mu)*(x + mu)./abs(x + mu).^3 ...
    - mu*(x - (1-mu))./abs(x - (1-mu)).^3;

% good initial guesses
delta = (mu/3)^(1/3);
x2 = 1 - mu;

x0_L1 = x2 - delta;
x0_L2 = x2 + delta;

% robust fzero calls
opts = optimset('Display','off');
xL1 = fzero(f, x0_L1, opts);
xL2 = fzero(f, x0_L2, opts);
end