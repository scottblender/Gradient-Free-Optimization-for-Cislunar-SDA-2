% ---- plot_jpl_orbit_catalog.m ---- %
clear; close all; clc;

% ---------------- Figure defaults ----------------
set(groot, ...
    'defaultAxesFontSize',34, ...
    'defaultAxesFontWeight','bold', ...
    'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontSize',30, ...
    'defaultTextFontWeight','bold', ...
    'defaultTextFontName','Times New Roman', ...
    'defaultLegendFontSize',26, ...
    'defaultLegendFontWeight','bold', ...
    'defaultLegendFontName','Times New Roman', ...
    'defaultAxesLabelFontSizeMultiplier',1.05, ...
    'defaultAxesTitleFontSizeMultiplier',1.0, ...
    'defaultLineLineWidth',3.0);

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

% Keep original row numbers so manual plot removal is stable
if ~ismember('rowID', T.Properties.VariableNames)
    T.rowID = (1:height(T))';
end

% ---------------- JPL constants ----------------
mu = 1.215058560962404E-2;
LU = 384400; % km
ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% -------------------------------------------------------------------------
% Plot-only database table.
%
% T is kept unchanged for the context/transfer figures and hard-coded orbit
% indices. Tplot is used only for the orbit-family database figures.
% -------------------------------------------------------------------------
Tplot = T;

% -------------------------------------------------------------------------
% MANUAL plot-only removal of specific bad orbits
%
% Step 1:
%   Set labelOrbitRowIDs = true and run the script.
%   The orbit-family plots will show rowID numbers on each plotted orbit.
%
% Step 2:
%   Read the rowID number beside the orbit you want to remove.
%
% Step 3:
%   Put that rowID in manualRemoveRowIDs and set labelOrbitRowIDs = false.
%
% Example:
%   manualRemoveRowIDs = [137 284];
% -------------------------------------------------------------------------
labelOrbitRowIDs = false;
manualRemoveRowIDs = [87, 88];

if ~isempty(manualRemoveRowIDs)
    removeMask = ismember(Tplot.rowID, manualRemoveRowIDs);

    fprintf('\nManually removing these orbit rowIDs from plots only:\n');
    disp(Tplot(removeMask, {'rowID','orbitFamily'}));

    Tplot(removeMask,:) = [];
end

% -------------------------------------------------------------------------
% OPTIONAL automatic plot-only removal of Gateway-like NRHO orbits, family
% by family. Leave this on if it helps; turn it off if you only want manual.
%
% This removes the most Gateway-like orbit from each derived NRHO family:
%   NNRHL1, NNRHL2, SNRHL1, SNRHL2
%
% It checks both the actual Lunar Gateway trajectory and its z-reflected
% version, so it can catch flipped Gateway-like orbits too.
% -------------------------------------------------------------------------
removeLGlikeNRHOForPlots = true;

% Increase to 2 if one still remains.
nRemovePerNRHFamily = 1;

targetNRHFamilies = ["NNRHL1", "NNRHL2", "SNRHL1", "SNRHL2"];

if removeLGlikeNRHOForPlots

    dt_lg_filter = 0.01;
    N_periods_filter = 1;

    s_lg_ic_filter = [1.02202108343387, 0, -0.182096487798513, ...
                      0, -0.103255420206012, 0]';

    T_lg_filter = 1.51110546287394;

    tspan_lg_filter = 0:dt_lg_filter:N_periods_filter*T_lg_filter;

    [~, s_lg_filter] = ode45(@(t,s) cr3bp_dynamics(t, s, mu), ...
                              tspan_lg_filter, s_lg_ic_filter, ode_opts);

    rLG = s_lg_filter(:,1:3);

    rLG_flip = rLG;
    rLG_flip(:,3) = -rLG_flip(:,3);

    famAll = string(Tplot.orbitFamily);

    allRemoveIdx = [];
    allDiag = table();

    for ff = 1:numel(targetNRHFamilies)

        fam = targetNRHFamilies(ff);
        candidateIdx = find(famAll == fam);

        if isempty(candidateIdx)
            warning('No candidate orbits found for %s.', fam);
            continue;
        end

        lgDirectScore = inf(numel(candidateIdx),1);
        lgFlipScore   = inf(numel(candidateIdx),1);
        lgSymScore    = inf(numel(candidateIdx),1);

        for jj = 1:numel(candidateIdx)

            ii = candidateIdx(jj);

            s = Tplot.state{ii};
            r = s(:,1:3);

            dminDirect = zeros(size(r,1),1);
            dminFlip   = zeros(size(r,1),1);

            for kk = 1:size(r,1)

                diffsDirect = rLG - r(kk,:);
                d2Direct = sum(diffsDirect.^2, 2);
                dminDirect(kk) = sqrt(min(d2Direct));

                diffsFlip = rLG_flip - r(kk,:);
                d2Flip = sum(diffsFlip.^2, 2);
                dminFlip(kk) = sqrt(min(d2Flip));

            end

            lgDirectScore(jj) = mean(dminDirect);
            lgFlipScore(jj)   = mean(dminFlip);
            lgSymScore(jj)    = min(lgDirectScore(jj), lgFlipScore(jj));

        end

        [~, ord] = sort(lgSymScore, 'ascend', 'MissingPlacement','last');

        nTake = min(nRemovePerNRHFamily, numel(candidateIdx));
        removeIdxFam = candidateIdx(ord(1:nTake));

        allRemoveIdx = [allRemoveIdx; removeIdxFam(:)];

        diagFam = table( ...
            Tplot.rowID(removeIdxFam), ...
            string(Tplot.orbitFamily(removeIdxFam)), ...
            lgDirectScore(ord(1:nTake)), ...
            lgFlipScore(ord(1:nTake)), ...
            lgSymScore(ord(1:nTake)), ...
            'VariableNames', {'rowID','orbitFamily','directLGScore','flippedLGScore','symmetricLGScore'});

        allDiag = [allDiag; diagFam];

    end

    allRemoveIdx = unique(allRemoveIdx);

    fprintf('\nAutomatically removing Gateway-like NRHO orbits from plots only, family by family:\n');
    disp(allDiag);

    Tplot(allRemoveIdx,:) = [];

end

% ---------------- Lagrange points ----------------
[xL1, xL2] = cr3bp_L1L2(mu);

% ---------------- Moon surface ----------------
R_moon = 1737.1 / LU; % Moon radius in LU

[Xm, Ym, Zm] = sphere(40);
Xm = R_moon * Xm + (1 - mu);
Ym = R_moon * Ym;
Zm = R_moon * Zm;

% ---------------- Catalog / orbit database ----------------
num_orbits      = height(T);
slots_per_orbit = 50;

tf     = T.('Period (TU) ');
states = T.('state');
times  = T.('time');

% Full catalog family labels
orbitFamily = string(T.orbitFamily);

% Plot-only family labels
orbitFamilyPlot = string(Tplot.orbitFamily);

orbit_database = cell(num_orbits, 1);
for i = 1:num_orbits
    t_raw  = times{i};
    s_raw  = states{i};
    period = tf(i);

    t_slots = orbit_slot_times(period, slots_per_orbit);

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

% ---------------- Colors ----------------
cL1   = [0 0 1];         % blue
cL2   = [1 0 0];         % red
cMoon = [0.70 0.70 0.70];
cLP   = [0.85 0.85 0.85];

% -------------------------------------------------------------------------
% Export / readability controls
%
% Keep ContentType = 'image' for faster Overleaf loading.
% The font sizes are intentionally large because each image is later scaled
% down in LaTeX as a subfigure.
% -------------------------------------------------------------------------
nPlotTotal_L1L2 = 16;
nPlotDRO        = 16;
maxPtsPerOrbit  = 300;
epsDPI          = 600;

figW = 10.5;
figH = 9;

axisFontSize   = 30;
labelFontSize  = 36;
legendFontSize = 34;
textFontSize   = 30;

markerSizeLP   = 13;
markerSizeMain = 12;
lineWidthMain  = 3.0;
lineWidthThick = 3.4;
lineWidthRef   = 1.8;

legendTokenSize = [28 16];

% ---------------- Family plotting loop ----------------
for i = 1:numel(families)

    fig = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    axis(ax,'equal');
    set(ax, 'TickLabelInterpreter','tex', 'Layer','top');

    % Reference-style view for the 3D family plots
    if i < 5
        ax.Projection = 'perspective';
        view(ax, -37.5, 30);
    else
        ax.Projection = 'orthographic';
        view(ax, 2);
    end

    current_pair = families{i};

    if i < 5

        famColors = [cL1; cL2];
        h = gobjects(5,1); % fam1, fam2, moon, L1, L2

        nPlotPerFamily = ceil(nPlotTotal_L1L2 / 2);

        for k = 1:2

            targetFamily = current_pair(k);
            idx = orbitFamilyPlot == targetFamily;
            subT = Tplot(idx, :);

            numOrbits = height(subT);

            if numOrbits > 0

                rows_to_plot = round(linspace(1, numOrbits, ...
                    min(nPlotPerFamily, numOrbits)));
                rows_to_plot = unique(rows_to_plot);

                fprintf('%s: plotting %d orbits from family %s\n', ...
                    filenames(i), numel(rows_to_plot), targetFamily);

                for rr = 1:numel(rows_to_plot)

                    r = rows_to_plot(rr);
                    state = subT.state{r};

                    plotStep = max(1, round(size(state,1) / maxPtsPerOrbit));
                    statePlot = state(1:plotStep:end,:);

                    p = plot3(ax, statePlot(:,1), statePlot(:,2), statePlot(:,3), '-', ...
                        'Color', famColors(k,:), 'LineWidth', lineWidthMain);

                    if labelOrbitRowIDs
                        midIdx = round(size(state,1)/2);
                        text(state(midIdx,1), state(midIdx,2), state(midIdx,3), ...
                            sprintf(' %d', subT.rowID(r)), ...
                            'FontSize', 16, ...
                            'FontWeight', 'bold', ...
                            'FontName', 'Times New Roman', ...
                            'Color', famColors(k,:), ...
                            'Parent', ax);
                    end

                    if rr == 1
                        h(k) = p;
                    end

                end

            else
                warning('No orbits found for family %s after plot-only filtering.', targetFamily);
            end

        end

        h(3) = surf(ax, Xm, Ym, Zm, ...
            'FaceColor', cMoon, ...
            'EdgeColor', 'none', ...
            'FaceLighting', 'gouraud');

        h(4) = plot3(ax, xL1, 0, 0, '^', ...
            'MarkerSize',markerSizeLP, ...
            'MarkerFaceColor',cLP, ...
            'MarkerEdgeColor',[0.60 0.60 0.60], ...
            'LineWidth',lineWidthRef);

        h(5) = plot3(ax, xL2, 0, 0, 'v', ...
            'MarkerSize',markerSizeLP, ...
            'MarkerFaceColor',cLP, ...
            'MarkerEdgeColor',[0.60 0.60 0.60], ...
            'LineWidth',lineWidthRef);

        labels = {'L1', 'L2', 'Moon', 'L1 point', 'L2 point'};

    else

        h = gobjects(2,1);

        targetFamily = current_pair(1);
        idx = orbitFamilyPlot == targetFamily;
        subT = Tplot(idx, :);

        yScale = 1;   % display-only vertical exaggeration for DRO figure

        numOrbits = height(subT);

        if numOrbits > 0

            rows_to_plot = round(linspace(1, numOrbits, ...
                min(nPlotDRO, numOrbits)));
            rows_to_plot = unique(rows_to_plot);

            fprintf('%s: plotting %d orbits from family %s\n', ...
                filenames(i), numel(rows_to_plot), targetFamily);

            for rr = 1:numel(rows_to_plot)

                r = rows_to_plot(rr);
                state = subT.state{r};

                plotStep = max(1, round(size(state,1) / maxPtsPerOrbit));
                statePlot = state(1:plotStep:end,:);

                p = plot(ax, statePlot(:,1), yScale*statePlot(:,2), '-', ...
                    'Color', cL1, 'LineWidth', lineWidthMain);

                if labelOrbitRowIDs
                    midIdx = round(size(state,1)/2);
                    text(state(midIdx,1), yScale*state(midIdx,2), ...
                        sprintf(' %d', subT.rowID(r)), ...
                        'FontSize', 16, ...
                        'FontWeight', 'bold', ...
                        'FontName', 'Times New Roman', ...
                        'Color', cL1, ...
                        'Parent', ax);
                end

                if rr == 1
                    h(1) = p;
                end

            end

        else
            warning('No orbits found for family %s after plot-only filtering.', targetFamily);
        end

        % Moon as a filled circle in the x-y plane
        th = linspace(0, 2*pi, 200);
        xMoon = (1 - mu) + R_moon*cos(th);
        yMoon = yScale * (R_moon*sin(th));

        h(2) = fill(ax, xMoon, yMoon, cMoon, ...
            'EdgeColor', 'none', ...
            'FaceAlpha', 1.0);

        labels = {'DRO', 'Moon'};

        % Switch to 2-D formatting for DROs
        view(ax, 2);
        axis(ax, 'equal');

        % Tighten limits to the actual DRO family
        x_center = 1 - mu;
        y_center = 0;

        dy = yScale * 0.015;
        padDRO = 1.15;
        dx = padDRO * dy;

        xlim(ax, [x_center - dx, x_center + dx]);
        ylim(ax, [y_center - dx, y_center + dx]);

        xlabel(ax, 'X (LU)');
        ylabel(ax, 'Y (LU)');
        zlabel(ax, '');

        ax.Box = 'on';
        ax.Layer = 'top';
        ax.YAxis.Exponent = 0;
        xtickformat(ax, '%.3f');
        ytickformat(ax, '%.2f');

    end

    xlabel(ax,'X (LU)');
    ylabel(ax,'Y (LU)');

    if i < 5
        zlabel(ax,'Z (LU)');
    else
        zlabel(ax,'');
    end

    format_axes_for_export(ax, axisFontSize, labelFontSize);

    camlight(ax, 'headlight');
    material(ax, 'dull');

    validHandles = isgraphics(h);
    if any(validHandles)
        lgd = legend(ax, h(validHandles), labels(validHandles));
        lgd.Location = 'northoutside';
        lgd.Orientation = 'horizontal';
        lgd.Box = 'on';
        lgd.FontSize = legendFontSize;
        lgd.FontWeight = 'bold';
        lgd.FontName = 'Times New Roman';
        lgd.ItemTokenSize = legendTokenSize;
    end

    % Do not override explicit DRO limits
    if i < 5
        axis(ax,'tight');
    end

    axis(ax,'vis3d');
    ax.Units = 'normalized';

    % Make axes occupy more of the image while leaving room for legend
    ax.Position = [0.10 0.13 0.84 0.70];
    ax.LooseInset = max(ax.TightInset, 0.02);

    save_eps_image(fig, FigDir, filenames(i) + ".eps", epsDPI);

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

% --- Colors ---
cCoast    = [0.91 0.29 0.24];
cTransfer = [0.27 0.31 0.86];
cOrbit    = [0.47 0.78 0.94];

fig = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
             'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

ax = axes(fig);
hold(ax,'on');
box(ax,'on');
axis(ax,'equal');
set(ax, 'TickLabelInterpreter','tex', 'Layer','top');
ax.Projection = 'orthographic';
view(ax, 32, 24);

% --- Trajectories ---
hTransfer = plot3(ax, s_transfer(:,1), s_transfer(:,2), s_transfer(:,3), '-', ...
    'LineWidth', lineWidthThick, 'Color', cTransfer);

hCoast = plot3(ax, s_lg(:,1), s_lg(:,2), s_lg(:,3), '-', ...
    'LineWidth', lineWidthThick, 'Color', cCoast);

plot3(ax, s_dep_orb(:,1), s_dep_orb(:,2), s_dep_orb(:,3), '-', ...
    'Color', cOrbit, 'LineWidth', lineWidthMain);

plot3(ax, s_arr_orb(:,1), s_arr_orb(:,2), s_arr_orb(:,3), '-', ...
    'Color', cOrbit, 'LineWidth', lineWidthMain);

% --- Markers / surface ---
hM = surf(ax, Xm, Ym, Zm, ...
    'FaceColor', cMoon, ...
    'EdgeColor', 'none', ...
    'FaceLighting', 'gouraud');

hL1 = plot3(ax, xL1, 0, 0, '^', ...
    'MarkerSize',markerSizeLP, ...
    'MarkerFaceColor',cLP, ...
    'MarkerEdgeColor',[0.6 0.6 0.6], ...
    'LineWidth',lineWidthRef);

hL2 = plot3(ax, xL2, 0, 0, 'v', ...
    'MarkerSize',markerSizeLP, ...
    'MarkerFaceColor',cLP, ...
    'MarkerEdgeColor',[0.6 0.6 0.6], ...
    'LineWidth',lineWidthRef);

hStart = plot3(ax, s_transfer(1,1), s_transfer(1,2), s_transfer(1,3), 'o', ...
    'MarkerSize',markerSizeMain, ...
    'MarkerFaceColor',cCoast, ...
    'MarkerEdgeColor','k', ...
    'LineWidth',lineWidthRef);

hEnd = plot3(ax, s_transfer(end,1), s_transfer(end,2), s_transfer(end,3), 's', ...
    'MarkerSize',markerSizeMain, ...
    'MarkerFaceColor',cTransfer, ...
    'MarkerEdgeColor','k', ...
    'LineWidth',lineWidthRef);

xlabel(ax,'x (LU)');
ylabel(ax,'y (LU)');
zlabel(ax,'z (LU)');

format_axes_for_export(ax, axisFontSize, labelFontSize);

camlight(ax, 'headlight');
material(ax, 'dull');

lgd = legend(ax, [hCoast, hTransfer, hStart, hEnd, hM, hL1, hL2], ...
    {'Coasting', 'Transfer', 'Start', 'End', 'Moon', 'L1', 'L2'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal');

lgd.Box = 'on';
lgd.FontSize = legendFontSize;
lgd.FontWeight = 'bold';
lgd.FontName = 'Times New Roman';
lgd.ItemTokenSize = legendTokenSize;

try
    lgd.NumColumns = 4;
catch
end

axis(ax,'tight');
axis(ax,'vis3d');

ax.Units = 'normalized';
ax.Position = [0.10 0.13 0.84 0.68];
ax.LooseInset = max(ax.TightInset, 0.02);

save_eps_image(fig, FigDir, 'lunar_gateway_low_thrust_context.eps', epsDPI);
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

    slot_times = orbit_slot_times(T_nh, slot_plot_count);
    [t_unique, idx_u] = unique(t_nh);
    s_unique = s_nh(idx_u,:);
    F_nh = griddedInterpolant(t_unique, s_unique, 'spline');
    s_slots = F_nh(slot_times);

    sampleSlot  = ceil(slot_plot_count / 3);
    sampleState = s_slots(sampleSlot,:);

    fig = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                 'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    axis(ax,'equal');
    set(ax, 'TickLabelInterpreter','tex', 'Layer','top');
    ax.Projection = 'orthographic';
    view(ax, 32, 24);

    % Downsample orbit curve for smaller EPS size.
    plotStepNH = max(1, round(size(s_nh,1) / maxPtsPerOrbit));
    s_nh_plot = s_nh(1:plotStepNH:end,:);

    hOrb = plot3(ax, s_nh_plot(:,1), s_nh_plot(:,2), s_nh_plot(:,3), '-', ...
        'Color', cTransfer, 'LineWidth', lineWidthMain);

    % Exclude the selected slot from the candidate-slot markers
    candidateMask = true(size(s_slots,1),1);
    candidateMask(sampleSlot) = false;

    hSlots = plot3(ax, ...
        s_slots(candidateMask,1), ...
        s_slots(candidateMask,2), ...
        s_slots(candidateMask,3), ...
        'o', ...
        'MarkerSize',8, ...
        'MarkerFaceColor','w', ...
        'MarkerEdgeColor',[0.20 0.20 0.20], ...
        'LineWidth',lineWidthRef);

       % Larger translucent marker underneath for shaded selected-slot effect
    scatter3(ax, ...
        sampleState(1), sampleState(2), sampleState(3), ...
        300, cCoast, ...
        'filled', ...
        'MarkerFaceAlpha', 0.25, ...
        'MarkerEdgeAlpha', 0.0);
    
    % Solid selected slot marker on top
    hSample = scatter3(ax, ...
        sampleState(1), sampleState(2), sampleState(3), ...
        120, cCoast, ...
        'filled', ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.8);
    
    % -----------------------------------------------------------------
    % Place "Selected slot" outside the orbit region and point to it
    % -----------------------------------------------------------------
    xL = xlim(ax);
    yL = ylim(ax);
    zL = zlim(ax);
    
    dx = xL(2) - xL(1);
    dy = yL(2) - yL(1);
    dz = zL(2) - zL(1);
    
    % Expand limits slightly so there is room for the label
    xlim(ax, [xL(1), xL(2) + 0.18*dx]);
    ylim(ax, [yL(1), yL(2) + 0.03*dy]);
    
    % Re-read limits after expanding
    xL = xlim(ax);
    yL = ylim(ax);
    zL = zlim(ax);
    
    % Label location (to the right of the plotted orbit)
    xText = xL(2) - 0.12*dx;
    yText = sampleState(2) + 0.02*dy;
    zText = sampleState(3) + 0.02*dz;
    
    % Arrow start point (slightly left of the text)
    xArrow = xText - 0.015*dx;
    yArrow = yText;
    zArrow = zText;
    
    % Draw label
    text(ax, xText, yText, zText, {'  Selected', '  slot'}, ...
        'FontSize', textFontSize, ...
        'FontWeight', 'bold', ...
        'FontName', 'Times New Roman', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', ...
        'Clipping', 'off');
    
    % Draw arrow to the selected slot
    quiver3(ax, ...
        xArrow, yArrow, zArrow, ...
        sampleState(1) - xArrow, ...
        sampleState(2) - yArrow, ...
        sampleState(3) - zArrow, ...
        0, ...
        'Color', 'k', ...
        'LineWidth', 1.8, ...
        'MaxHeadSize', 0.6);

    hM = surf(ax, Xm, Ym, Zm, ...
        'FaceColor', cMoon, ...
        'EdgeColor', 'none', ...
        'FaceLighting', 'gouraud');

    hL1s = plot3(ax, xL1, 0, 0, '^', ...
        'MarkerSize',markerSizeLP, ...
        'MarkerFaceColor',cLP, ...
        'MarkerEdgeColor',[0.6 0.6 0.6], ...
        'LineWidth',lineWidthRef);

    xlabel(ax,'x (LU)');
    ylabel(ax,'y (LU)');
    zlabel(ax,'z (LU)');

    format_axes_for_export(ax, axisFontSize, labelFontSize);

    camlight(ax, 'headlight');
    material(ax, 'dull');

    lgd = legend(ax, [hOrb; hSlots; hSample; hM; hL1s], ...
    {'Halo orbit', 'Candidate slots', 'Selected slot', 'Moon', 'L1'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal');

    lgd.Box = 'on';
    lgd.FontSize = legendFontSize;
    lgd.FontWeight = 'bold';
    lgd.FontName = 'Times New Roman';
    lgd.ItemTokenSize = legendTokenSize;
    
    try
        lgd.NumColumns = 3;
    catch
    end
    
    % Manually center the legend above the axes
    drawnow;
    lgd.Units = 'normalized';
    
    legendWidth  = 0.82;
    legendHeight = lgd.Position(4);
    
    lgd.Position = [ ...
        0.5 - legendWidth/2, ...  % centered horizontally in figure
        0.86, ...                 % vertical location near top
        legendWidth, ...
        legendHeight];

    axis(ax,'tight');
    axis(ax,'vis3d');

    ax.Units = 'normalized';
    ax.Position = [0.10 0.13 0.72 0.68];
    ax.LooseInset = max(ax.TightInset, 0.02);

    save_eps_image(fig, FigDir, 'northern_halo_slot_discretization.eps', epsDPI);
    close(fig);

end

% ---------------- Helper functions ----------------
function save_eps_image(fig, outDir, fileName, dpi)
    outPath = fullfile(outDir, char(string(fileName)));

    % Keep raster/image export for Overleaf speed.
    % Large MATLAB fonts + 600 DPI keeps labels readable after scaling.
    exportgraphics(fig, outPath, ...
        'ContentType', 'image', ...
        'Resolution', dpi);

    fprintf('Saved %s\n', outPath);
end

function format_axes_for_export(ax, axisFontSize, labelFontSize)

    set(ax, ...
        'FontSize', axisFontSize, ...
        'FontWeight', 'bold', ...
        'FontName', 'Times New Roman', ...
        'LineWidth', 2.4);

    ax.XLabel.FontSize = labelFontSize;
    ax.YLabel.FontSize = labelFontSize;
    ax.ZLabel.FontSize = labelFontSize;

    ax.XLabel.FontWeight = 'bold';
    ax.YLabel.FontWeight = 'bold';
    ax.ZLabel.FontWeight = 'bold';

    ax.XLabel.FontName = 'Times New Roman';
    ax.YLabel.FontName = 'Times New Roman';
    ax.ZLabel.FontName = 'Times New Roman';

    ax.LooseInset = max(ax.TightInset, 0.02);

end

function [xL1, xL2] = cr3bp_L1L2(mu)
    f = @(x) x ...
        - (1-mu) * (x + mu) ./ abs(x + mu).^3 ...
        - mu     * (x - (1-mu)) ./ abs(x - (1-mu)).^3;

    delta = (mu/3)^(1/3);

    xL1 = fzero(f, [1-mu-delta, 1-mu-1e-6]);
    xL2 = fzero(f, [1-mu+1e-6, 1-mu+delta+0.5]);
end
