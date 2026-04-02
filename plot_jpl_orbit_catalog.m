% ---- plot_jpl_orbit_catalog.m ---- %
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

% ---------------- Paths / output directory ----------------
thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);
addpath(genpath(thisDir));

FigDir = fullfile(thisDir, 'database_figs');
if ~exist(FigDir, 'dir')
    mkdir(FigDir);
end

fprintf('Figure directory: %s\n', FigDir);

% ---------------- Load JPL data ----------------
catalogPath = fullfile(thisDir, 'JPL_CR3BP_OrbitCatalog.mat');
if ~isfile(catalogPath)
    error('Data file not found: %s', catalogPath);
end

S = load(catalogPath);
T = S.T;

% ---------------- JPL constants ----------------
mu = 1.215058560962404E-2;
ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% ---------------- Lagrange points ----------------
[xL1, xL2] = cr3bp_L1L2(mu);

% ---------------- Catalog / orbit database ----------------
num_orbits      = height(T);
slots_per_orbit = 50;

tf     = T.('Period (TU) ');
states = T.('state');
times  = T.('time');

% normalize family column so string comparisons are robust
orbitFamily = string(T.orbitFamily);

orbit_database = cell(num_orbits, 1);
for i = 1:num_orbits
    t_raw  = times{i};
    s_raw  = states{i};
    period = tf(i);

    t_slots = linspace(0, period, slots_per_orbit)';

    [t_unique, idx_u] = unique(t_raw);
    s_unique = s_raw(idx_u, :);
    F        = griddedInterpolant(t_unique, s_unique, 'spline');

    orbit_database{i} = F(t_slots);
end

% ---------------- Family definitions ----------------
families = {
    ["NHL1",   "NHL2"], ...
    ["SHL1",   "SHL2"], ...
    ["NNRHL1", "NNRHL2"], ...
    ["SNRHL1", "SNRHL2"], ...
    ["DRO"]
};

filenames = ["northern_halo", "southern_halo", ...
             "northern_rectilinear", "southern_rectilinear", ...
             "dro_family"];

% ---------------- Family plotting loop ----------------
for i = 1:numel(families)
    fig = figure('Color','w','Units','inches','Position',[1 1 8 6], ...
                 'PaperUnits','inches','PaperPosition',[0 0 8 6]);

    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'on');
    axis(ax,'equal');
    set(ax, 'TickLabelInterpreter','tex', 'Layer','top');
    ax.Projection = 'orthographic';

    if i <= 2
        view(ax, -35, 30);
    elseif i <= 4
        view(ax, -50, 25);
    else
        view(ax, -35, 20);
    end

    current_pair = families{i};

    if i < 5
        famColors = [0.00 0.45 0.74; 0.85 0.33 0.10];
        h = gobjects(5,1); % fam1, fam2, moon, L1, L2

        for k = 1:2
            targetFamily = current_pair(k);
            idx = orbitFamily == targetFamily;
            subT = T(idx, :);

            numOrbits = height(subT);
            if numOrbits > 0
                plot_stride = max(1, round(numOrbits / 15));
                rows_to_plot = 1:plot_stride:numOrbits;

                for r = rows_to_plot
                    state = subT.state{r};
                    p = plot3(ax, state(:,1), state(:,2), state(:,3), '-', ...
                        'Color', famColors(k,:), 'LineWidth', 2.0);

                    if r == rows_to_plot(1)
                        h(k) = p;
                    end
                end
            end
        end

        h(3) = plot3(ax, 1-mu, 0, 0, 'o', ...
            'MarkerSize',8, ...
            'MarkerFaceColor',[0.70 0.70 0.70], ...
            'MarkerEdgeColor',[0.30 0.30 0.30], ...
            'LineWidth',1.0);

        h(4) = plot3(ax, xL1, 0, 0, '^', ...
            'MarkerSize',8, ...
            'MarkerFaceColor',[0.85 0.85 0.85], ...
            'MarkerEdgeColor',[0.60 0.60 0.60], ...
            'LineWidth',1.0);

        h(5) = plot3(ax, xL2, 0, 0, 'v', ...
            'MarkerSize',8, ...
            'MarkerFaceColor',[0.85 0.85 0.85], ...
            'MarkerEdgeColor',[0.60 0.60 0.60], ...
            'LineWidth',1.0);

        labels = {'L1 family', 'L2 family', 'Moon', 'L1', 'L2'};
    else
        h = gobjects(2,1);

        targetFamily = current_pair(1);
        idx = orbitFamily == targetFamily;
        subT = T(idx, :);

        numOrbits = height(subT);
        if numOrbits > 0
            plot_stride = max(1, round(numOrbits / 15));
            rows_to_plot = 1:plot_stride:numOrbits;

            for r = rows_to_plot
                state = subT.state{r};
                p = plot3(ax, state(:,1), state(:,2), state(:,3), '-', ...
                    'Color', [0.00 0.45 0.74], 'LineWidth', 2.0);

                if r == rows_to_plot(1)
                    h(1) = p;
                end
            end
        end

        h(2) = plot3(ax, 1-mu, 0, 0, 'o', ...
            'MarkerSize',8, ...
            'MarkerFaceColor',[0.70 0.70 0.70], ...
            'MarkerEdgeColor',[0.30 0.30 0.30], ...
            'LineWidth',1.0);

        labels = {'DRO', 'Moon'};
    end

    xlabel(ax,'x (LU)');
    ylabel(ax,'y (LU)');
    zlabel(ax,'z (LU)');
    set(ax,'FontSize',16,'LineWidth',1.8);

    validHandles = isgraphics(h);
    if any(validHandles)
        lgd = legend(ax, h(validHandles), labels(validHandles), 'Location','best');
        lgd.Box = 'on';
        lgd.ItemTokenSize = [18 12];
    end

    axis(ax,'tight');
    axis(ax,'vis3d');
    pad = 0.10;
    ax.Position = [pad pad 1-2*pad 1-2*pad];

    save_pdf(fig, FigDir, filenames(i) + ".pdf");
    close(fig);
end

% ---------------- Lunar Gateway + low-thrust context figure ----------------

% --- Lunar Gateway truth propagated with cr3bp_dynamics ---
dt_lg     = 0.001;
N_periods = 1;
s_lg_ic   = [1.02202108343387, 0, -0.182096487798513, ...
             0, -0.103255420206012, 0]';
T_lg      = 1.51110546287394;

tspan_lg = 0:dt_lg:N_periods*T_lg;
[~, s_lg] = ode45(@(t,s) cr3bp_dynamics(t, s, mu), tspan_lg, s_lg_ic, ode_opts);

% --- Transfer settings ---
missionCfg = struct();
missionCfg.type = "LOW_THRUST_TRANSFER";
missionCfg.transfer.depOrbitIndex = 52;
missionCfg.transfer.depSlot       = 10;
missionCfg.transfer.arrOrbitIndex = 400;
missionCfg.transfer.arrSlot       = 1;
missionCfg.transfer.dt            = 0.001;
missionCfg.transfer.solverMode    = "LOW_THRUST_CLASS";

missionCfg.transfer.lowthrust.sigma           = 1.0;
missionCfg.transfer.lowthrust.m0              = 1.0;
missionCfg.transfer.lowthrust.Tmax            = 0.3672;
missionCfg.transfer.lowthrust.ve              = 39.8;
missionCfg.transfer.lowthrust.tf_guess        = 2.0;
missionCfg.transfer.lowthrust.tf_lb           = 0.1;
missionCfg.transfer.lowthrust.tf_ub           = 12.0;
missionCfg.transfer.lowthrust.lambda_guess    = [-0.25; 0.75; 0.35; -0.20; 0.40; 0.10; 0.05];
missionCfg.transfer.lowthrust.lambda_lb       = -20 * ones(7,1);
missionCfg.transfer.lowthrust.lambda_ub       =  20 * ones(7,1);
missionCfg.transfer.lowthrust.w_pos_indirect  = 1;
missionCfg.transfer.lowthrust.w_vel_indirect  = 1;
missionCfg.transfer.lowthrust.w_norm_indirect = 1;
missionCfg.transfer.lowthrust.w_mass_indirect = 1;

% --- Build low-thrust truth trajectory ---
[t_transfer, s_transfer, truthInfo] = build_target_truth( ...
    missionCfg, T, orbit_database, times, states, mu, ode_opts); 

depIdx  = missionCfg.transfer.depOrbitIndex;
arrIdx  = missionCfg.transfer.arrOrbitIndex;
depSlot = missionCfg.transfer.depSlot;
arrSlot = missionCfg.transfer.arrSlot;

depSlot = max(1, min(depSlot, size(orbit_database{depIdx},1)));
arrSlot = max(1, min(arrSlot, size(orbit_database{arrIdx},1)));

s_dep_orb = states{depIdx};
s_arr_orb = states{arrIdx};
depState0 = orbit_database{depIdx}(depSlot,:); 
arrState0 = orbit_database{arrIdx}(arrSlot,:); 

% --- Colors (matched to your prior figure) ---
cCoast    = [0.91 0.29 0.24];
cTransfer = [0.27 0.31 0.86];
cOrbit    = [0.47 0.78 0.94];
cMoon     = [0.55 0.58 0.62];
cLP       = [0.88 0.88 0.88];

fig = figure('Color','w','Units','inches','Position',[1 1 8 6], ...
             'PaperUnits','inches','PaperPosition',[0 0 8 6]);

ax = axes(fig);
hold(ax,'on');
box(ax,'on');
grid(ax,'on');
axis(ax,'equal');
set(ax, 'TickLabelInterpreter','tex', 'Layer','top');
ax.Projection = 'orthographic';
view(ax, 32, 24);

% --- Trajectories ---
hTransfer = plot3(ax, s_transfer(:,1), s_transfer(:,2), s_transfer(:,3), '-', ...
    'LineWidth', 2.4, 'Color', cTransfer);

hCoast = plot3(ax, s_lg(:,1), s_lg(:,2), s_lg(:,3), '-', ...
    'LineWidth', 2.4, 'Color', cCoast);

plot3(ax, s_dep_orb(:,1), s_dep_orb(:,2), s_dep_orb(:,3), '-', ...
    'Color', cOrbit, 'LineWidth', 1.8);

plot3(ax, s_arr_orb(:,1), s_arr_orb(:,2), s_arr_orb(:,3), '-', ...
    'Color', cOrbit, 'LineWidth', 1.8);

% --- Markers ---
hM = plot3(ax, 1-mu, 0, 0, 'o', ...
    'MarkerSize',8, ...
    'MarkerFaceColor',cMoon, ...
    'MarkerEdgeColor',cMoon, ...
    'LineWidth',1.0);

hL1 = plot3(ax, xL1, 0, 0, '^', ...
    'MarkerSize',8, ...
    'MarkerFaceColor',cLP, ...
    'MarkerEdgeColor',[0.6 0.6 0.6], ...
    'LineWidth',1.0);

hL2 = plot3(ax, xL2, 0, 0, 'v', ...
    'MarkerSize',8, ...
    'MarkerFaceColor',cLP, ...
    'MarkerEdgeColor',[0.6 0.6 0.6], ...
    'LineWidth',1.0);

hStart = plot3(ax, s_transfer(1,1), s_transfer(1,2), s_transfer(1,3), 'o', ...
    'MarkerSize',9, ...
    'MarkerFaceColor',cCoast, ...
    'MarkerEdgeColor','k', ...
    'LineWidth',1.0);

hEnd = plot3(ax, s_transfer(end,1), s_transfer(end,2), s_transfer(end,3), 's', ...
    'MarkerSize',9, ...
    'MarkerFaceColor',cTransfer, ...
    'MarkerEdgeColor','k', ...
    'LineWidth',1.0);

xlabel(ax,'x (LU)');
ylabel(ax,'y (LU)');
zlabel(ax,'z (LU)');
set(ax,'FontSize',16,'LineWidth',1.8);

lgd = legend(ax, [hCoast, hTransfer, hStart, hEnd, hM, hL1, hL2], ...
    {'Coasting', 'Transfer', 'Transfer start', 'Transfer end', 'Moon', 'L1', 'L2'}, ...
    'Location', 'northeast');
lgd.Box = 'on';
lgd.ItemTokenSize = [18 12];

axis(ax,'tight');
axis(ax,'vis3d');
pad = 0.10;
ax.Position = [pad pad 1-2*pad 1-2*pad];

save_pdf(fig, FigDir, 'lunar_gateway_low_thrust_context.pdf');
close(fig);

% ---------------- Northern halo slot discretization figure ----------------
idxNH = find(orbitFamily == "NHL1", 1, 'first');
if isempty(idxNH)
    idxNH = find(orbitFamily == "NHL2", 1, 'first');
end

slot_plot_count = 20;

if isempty(idxNH)
    warning('No northern halo orbit found for slot discretization plot.');
else
    s_nh = states{idxNH};
    t_nh = times{idxNH};
    T_nh = tf(idxNH);

    slot_times = linspace(0, T_nh, slot_plot_count)';
    [t_unique, idx_u] = unique(t_nh);
    s_unique = s_nh(idx_u,:);
    F_nh = griddedInterpolant(t_unique, s_unique, 'spline');
    s_slots = F_nh(slot_times);

    sampleSlot  = ceil(slot_plot_count / 3);
    sampleState = s_slots(sampleSlot,:);

    fig = figure('Color','w','Units','inches','Position',[1 1 8 6], ...
                 'PaperUnits','inches','PaperPosition',[0 0 8 6]);

    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'on');
    axis(ax,'equal');
    set(ax, 'TickLabelInterpreter','tex', 'Layer','top');
    ax.Projection = 'orthographic';
    view(ax, 32, 24);

    hOrb = plot3(ax, s_nh(:,1), s_nh(:,2), s_nh(:,3), '-', ...
        'Color', cTransfer, 'LineWidth', 2.0);

    hSlots = plot3(ax, s_slots(:,1), s_slots(:,2), s_slots(:,3), 'o', ...
        'MarkerSize',5, ...
        'MarkerFaceColor','w', ...
        'MarkerEdgeColor',[0.20 0.20 0.20], ...
        'LineWidth',1.0);

    hSample = plot3(ax, sampleState(1), sampleState(2), sampleState(3), 'o', ...
        'MarkerSize',8, ...
        'MarkerFaceColor',cCoast, ...
        'MarkerEdgeColor','k', ...
        'LineWidth',1.0);

    text(sampleState(1), sampleState(2), sampleState(3), '  selected slot', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Parent', ax);

    hM = plot3(ax, 1-mu, 0, 0, 'o', ...
        'MarkerSize',8, ...
        'MarkerFaceColor',cMoon, ...
        'MarkerEdgeColor',cMoon, ...
        'LineWidth',1.0);

    hL1s = plot3(ax, xL1, 0, 0, '^', ...
        'MarkerSize',8, ...
        'MarkerFaceColor',cLP, ...
        'MarkerEdgeColor',[0.6 0.6 0.6], ...
        'LineWidth',1.0);

    xlabel(ax,'x (LU)');
    ylabel(ax,'y (LU)');
    zlabel(ax,'z (LU)');
    set(ax,'FontSize',16,'LineWidth',1.8);

    lgd = legend(ax, [hOrb; hSlots; hSample; hM; hL1s], ...
        {'Northern halo orbit', 'Candidate slots', 'Selected slot example', 'Moon', 'L1'}, ...
        'Location', 'best');
    lgd.Box = 'on';
    lgd.ItemTokenSize = [18 12];

    axis(ax,'tight');
    axis(ax,'vis3d');
    pad = 0.10;
    ax.Position = [pad pad 1-2*pad 1-2*pad];

    save_pdf(fig, FigDir, 'northern_halo_slot_discretization.pdf');
    close(fig);
end

% ---------------- Helper functions ----------------
function save_pdf(fig, outDir, fileName)
    outPath = fullfile(outDir, char(string(fileName)));
    exportgraphics(fig, outPath, 'ContentType', 'vector');
    fprintf('Saved %s\n', outPath);
end

function [xL1, xL2] = cr3bp_L1L2(mu)
    f = @(x) x ...
        - (1-mu) * (x + mu) ./ abs(x + mu).^3 ...
        - mu     * (x - (1-mu)) ./ abs(x - (1-mu)).^3;

    delta = (mu/3)^(1/3);
    x2 = 1 - mu;

    x0_L1 = x2 - delta;
    x0_L2 = x2 + delta;

    opts = optimset('Display','off');
    xL1 = fzero(f, x0_L1, opts);
    xL2 = fzero(f, x0_L2, opts);
end