function details = plot_reviewer2_constellation_geometry(selection,figureDir,stemPrefix,saveFigures)
%PLOT_REVIEWER2_CONSTELLATION_GEOMETRY Plot selected observer constellations.
%
% Each row in selection is one manuscript panel. Each unique selected
% periodic observer orbit is propagated for one period and drawn once as a
% solid line; individual observer phase locations are retained as markers.
% This avoids renderer z-fighting when multiple observers occupy different
% slots on the same periodic orbit.
%
% Target trajectories use the same mission colors as the tracking-case
% introduction figures. Earth is intentionally omitted so the result panels
% stay focused on the lunar-region constellation geometry. Low-thrust panels
% also show the departure/arrival periodic orbits and transfer endpoints.
%
% Required selection columns:
%   Mission, PanelKey, PanelLabel, RunFile, BestObjective
%
% The function exports one EPS/PNG panel per row so LaTeX can arrange the
% panels as subfigures. It also returns the selected orbit/slot/family data.

if nargin < 4 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
assert(istable(selection),'selection must be a table.');
required = ["Mission","PanelKey","PanelLabel","RunFile","BestObjective"];
assert(all(ismember(required,string(selection.Properties.VariableNames))), ...
    'Geometry selection table is missing required columns.');

figureDir = string(figureDir);
stemPrefix = string(stemPrefix);
if saveFigures
    assert(strlength(figureDir) > 0,'Figure directory is empty.');
    if ~isfolder(figureDir), mkdir(figureDir); end
end

details = table();
for k = 1:height(selection)
    runFile = string(selection.RunFile(k));
    trackingFile = string(fullfile(fileparts(runFile),'tracking_data.mat'));
    assert(isfile(runFile) && isfile(trackingFile), ...
        'Missing selected run/tracking file for geometry panel.');

    S = load(runFile,'runState');
    T = load(trackingFile,'tracking');
    r = S.runState;
    tracking = T.tracking;
    assert(isfield(r,'observers') && istable(r.observers) && ...
        height(r.observers) >= 1, ...
        'Selected run does not contain observer solution data.');

    fig = make_geometry_figure(r,tracking);
    panelStem = stemPrefix + "_" + mission_code(selection.Mission(k)) + ...
        "_" + sanitize_token(selection.PanelKey(k));
    if saveFigures
        export_geometry_figure(fig,fullfile(figureDir,panelStem));
    end

    families = strjoin(string(r.observers.orbit_family),";");
    orbitIndices = strjoin(string(r.observers.orbit_index),";");
    slotIndices = strjoin(string(r.observers.slot_index),";");
    row = table(string(selection.Mission(k)),string(selection.PanelKey(k)), ...
        string(selection.PanelLabel(k)),double(selection.BestObjective(k)), ...
        string(runFile),height(r.observers),families,orbitIndices,slotIndices, ...
        string(panelStem), ...
        'VariableNames',{'Mission','PanelKey','PanelLabel','BestObjective', ...
        'RunFile','NumObservers','OrbitFamilies','OrbitIndices','SlotIndices', ...
        'FigureStem'});
    details = [details;row]; %#ok<AGROW>
end
end


function fig = make_geometry_figure(runState,tracking)
mu = runState.settings.mu;
LU = runState.settings.LU;
mission = string(runState.settings.mission.type);
truth = tracking.truth(:,1:3);
observers = runState.observers;
nObs = height(observers);
targetColor = reviewer2_target_color(mission);

% Use a slightly larger canvas than the introductory plots while retaining
% their centered 3-D presentation. The extra physical margin is deliberate:
% projected x/y/z labels can extend beyond a perspective axes Position.
fig = figure('Color','w','Units','inches','Position',[1 1 8.2 7.6], ...
    'PaperUnits','inches','PaperSize',[8.2 7.6], ...
    'PaperPosition',[0 0 8.2 7.6],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
plotPosition = [0.18 0.19 0.64 0.60];
ax = axes(fig,'Units','normalized','Position',plotPosition);
ax.PositionConstraint = 'innerposition';
hold(ax,'on'); box(ax,'on'); axis(ax,'equal');

hTarget = plot3(ax,truth(:,1),truth(:,2),truth(:,3),'-', ...
    'Color',targetColor,'LineWidth',2.8,'DisplayName','Target trajectory');

% Draw each distinct periodic observer orbit once. Multiple observers on the
% same family/orbit index differ only by slot/phase, so overdrawing the same
% 3-D curve can create a dashed-looking z-buffer artifact. Their individual
% phase markers are still all shown below.
family = string(observers.orbit_family);
orbitIndex = string(observers.orbit_index);
orbitKeys = family + "_" + orbitIndex;
uniqueOrbitKeys = unique(orbitKeys,'stable');
observerColors = lines(max(numel(uniqueOrbitKeys),1));
allObserverPoints = zeros(0,3);
hObserver = gobjects(1,1);
opts = odeset('RelTol',1e-11,'AbsTol',1e-12);
observerLineWidth = 1.65;
if mission == "LOW_THRUST_TRANSFER", observerLineWidth = 1.45; end

for u = 1:numel(uniqueOrbitKeys)
    member = find(orbitKeys == uniqueOrbitKeys(u),1,'first');
    period = double(observers.period_TU(member));
    validateattributes(period,{'numeric'},{'scalar','real','finite','positive'});
    tPlot = linspace(0,period,300);
    initialState = observers.initial_state(member,:)';
    [~,state] = ode45(@(t,s) cr3bp_dynamics(t,s,mu),tPlot,initialState,opts);
    allObserverPoints = [allObserverPoints;state(:,1:3)]; %#ok<AGROW>
    h = plot3(ax,state(:,1),state(:,2),state(:,3), ...
        'LineStyle','-','Color',observerColors(u,:), ...
        'LineWidth',observerLineWidth,'HandleVisibility','off');
    if u == 1, hObserver = h; end
end
set(hObserver,'HandleVisibility','on','DisplayName','Observer orbits');

for j = 1:nObs
    u = find(uniqueOrbitKeys == orbitKeys(j),1,'first');
    phaseState = observers.initial_state(j,:);
    plot3(ax,phaseState(1),phaseState(2),phaseState(3),'o', ...
        'MarkerSize',5.5,'MarkerFaceColor',observerColors(u,:), ...
        'MarkerEdgeColor','k','LineWidth',0.7,'HandleVisibility','off');
end

% Low-thrust panels retain the same endpoint-orbit context as the
% introductory target-case figure, while the optimized observer orbits
% remain the main comparison quantity.
endpointPoints = zeros(0,3);
hEndpoint = gobjects(0);
hStart = gobjects(0);
hEnd = gobjects(0);
if mission == "LOW_THRUST_TRANSFER"
    assert(size(tracking.truth,2) >= 6, ...
        'Low-thrust geometry requires six-component saved truth states.');
    [departureOrbit,arrivalOrbit] = low_thrust_endpoint_orbits( ...
        tracking.truth(1,1:6),tracking.truth(end,1:6));
    cReference = [0.65 0.65 0.65];
    hEndpoint = plot3(ax,departureOrbit(:,1),departureOrbit(:,2), ...
        departureOrbit(:,3),'-','Color',cReference,'LineWidth',1.0, ...
        'DisplayName','Endpoint orbits');
    plot3(ax,arrivalOrbit(:,1),arrivalOrbit(:,2),arrivalOrbit(:,3),'-', ...
        'Color',cReference,'LineWidth',1.0,'HandleVisibility','off');
    endpointPoints = [departureOrbit(:,1:3);arrivalOrbit(:,1:3)];

    hStart = plot3(ax,truth(1,1),truth(1,2),truth(1,3),'o', ...
        'MarkerSize',8,'MarkerFaceColor',reviewer2_target_color("LUNAR_GATEWAY"), ...
        'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','Start');
    hEnd = plot3(ax,truth(end,1),truth(end,2),truth(end,3),'s', ...
        'MarkerSize',8,'MarkerFaceColor',targetColor, ...
        'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','End');
end

moonCenter = [1-mu,0,0];
moonRadius = 1737.1/LU;
[sx,sy,sz] = sphere(30);
hMoon = surf(ax,moonCenter(1)+moonRadius*sx, ...
    moonCenter(2)+moonRadius*sy,moonCenter(3)+moonRadius*sz, ...
    'FaceColor',[0.72 0.72 0.72],'EdgeColor','none', ...
    'FaceLighting','gouraud','DisplayName','Moon');
camlight(ax,'headlight'); material(ax,'dull');

[xL1,xL2] = collinear_lagrange_points(mu);
hL1 = plot3(ax,xL1,0,0,'^','MarkerSize',8, ...
    'MarkerFaceColor',[0.82 0.82 0.82],'MarkerEdgeColor','k', ...
    'LineWidth',1.0,'DisplayName','L1');
hL2 = plot3(ax,xL2,0,0,'v','MarkerSize',8, ...
    'MarkerFaceColor',[0.82 0.82 0.82],'MarkerEdgeColor','k', ...
    'LineWidth',1.0,'DisplayName','L2');

% Size the result panel from lunar-region geometry only. Earth remains
% intentionally absent from both the drawing and limits.
allPoints = [truth;allObserverPoints;endpointPoints;moonCenter;xL1 0 0;xL2 0 0];
xlim(ax,padded_limits(allPoints(:,1),0.10));
ylim(ax,padded_limits(allPoints(:,2),0.12));
zlim(ax,padded_limits(allPoints(:,3),0.12));
axis(ax,'vis3d');
ax.Projection = 'perspective';
view(ax,-37.5,30);
grid(ax,'off');

xlabel(ax,'x (LU)','FontWeight','bold');
ylabel(ax,'y (LU)','FontWeight','bold');
zlabel(ax,'z (LU)','FontWeight','bold');
set(ax,'FontName','Times New Roman','FontSize',12,'FontWeight','bold', ...
    'LineWidth',1.2,'Layer','top');
ax.XLabel.FontSize = 14; ax.YLabel.FontSize = 14; ax.ZLabel.FontSize = 14;

if mission == "LOW_THRUST_TRANSFER"
    legendHandles = [hEndpoint hTarget hObserver hStart hEnd hMoon hL1 hL2];
    legendLabels = {'Endpoint orbits','Target trajectory','Observer orbits', ...
        'Start','End','Moon','L1','L2'};
    numColumns = 4;
else
    legendHandles = [hTarget hObserver hMoon hL1 hL2];
    legendLabels = {'Target trajectory','Observer orbits','Moon','L1','L2'};
    numColumns = 3;
end
lgd = legend(ax,legendHandles,legendLabels, ...
    'Orientation','horizontal','NumColumns',numColumns,'Box','on');
lgd.FontName = 'Times New Roman';
lgd.FontSize = 12;
lgd.FontWeight = 'bold';
lgd.ItemTokenSize = [18 10];
lgd.Units = 'normalized';
finalize_centered_geometry_axes(ax,lgd,plotPosition);
end


function finalize_centered_geometry_axes(ax,lgd,plotPosition)
% Keep the complete perspective axes and legend centered inside the canvas.
%
% The legend is given a fixed centered strip instead of allowing MATLAB to
% resize the perspective axes. The axes Position is then restored explicitly,
% matching the construction used by the introductory tracking-case figures.
axis(ax,'vis3d');
ax.PositionConstraint = 'innerposition';
lgd.Units = 'normalized';
drawnow;

legendPosition = lgd.Position;
legendPosition(1) = 0.5-legendPosition(3)/2;
legendPosition(2) = 0.835;
lgd.Position = legendPosition;
lgd.AutoUpdate = 'off';

ax.Position = plotPosition;
drawnow;

% Preserve generous export padding around projected labels. LooseInset is
% applied after the final camera/legend layout so MATLAB cannot shrink the
% 3-D box to make room for the legend.
tightInset = ax.TightInset;
minInset = [0.035 0.045 0.025 0.025];
ax.LooseInset = max(tightInset,minInset);
ax.Position = plotPosition;
drawnow;
end


function [departureOrbit,arrivalOrbit] = low_thrust_endpoint_orbits(startState,endState)
% Recover the full periodic orbits containing the fixed LT endpoint states.
persistent cachedStart cachedEnd cachedDeparture cachedArrival
startState = double(startState(:).');
endState = double(endState(:).');
if ~isempty(cachedStart) && isequal(size(cachedStart),size(startState)) && ...
        max(abs(cachedStart-startState)) < 1e-12 && ...
        max(abs(cachedEnd-endState)) < 1e-12
    departureOrbit = cachedDeparture;
    arrivalOrbit = cachedArrival;
    return;
end

paths = setup_project();
catalog = load(paths.catalog,'T');
departureOrbit = find_reference_orbit_for_state(catalog.T,startState);
arrivalOrbit = find_reference_orbit_for_state(catalog.T,endState);
cachedStart = startState;
cachedEnd = endState;
cachedDeparture = departureOrbit;
cachedArrival = arrivalOrbit;
end


function orbitState = find_reference_orbit_for_state(T,targetState)
% Phase-independent catalog lookup used only for LT endpoint visualization.
assert(istable(T) && ismember('state',T.Properties.VariableNames), ...
    'Observer catalog must contain the state trajectory column.');
targetState = targetState(:).';
assert(numel(targetState)==6 && all(isfinite(targetState)), ...
    'Reference state must contain six finite CR3BP components.');

bestError = inf;
bestOrbit = [];
for k = 1:height(T)
    state = T.state{k};
    if isempty(state) || size(state,2)<6, continue; end
    state6 = state(:,1:6);
    state6 = state6(all(isfinite(state6),2),:);
    if isempty(state6), continue; end
    thisError = min(vecnorm(state6-targetState,2,2));
    if thisError < bestError
        bestError = thisError;
        bestOrbit = state(:,1:6);
    end
end
assert(~isempty(bestOrbit) && isfinite(bestError), ...
    'Could not identify an LT endpoint reference orbit.');
assert(bestError < 2.5e-2, ...
    ['LT endpoint does not match the observer catalog closely enough for ' ...
     'reference-orbit plotting (minimum state error %.6e).'],bestError);
orbitState = bestOrbit;
end


function [xL1,xL2] = collinear_lagrange_points(mu)
equilibrium = @(x) x ...
    -(1-mu)*(x+mu)./abs(x+mu).^3 ...
    -mu*(x-1+mu)./abs(x-1+mu).^3;
xL1 = fzero(equilibrium,1-mu-0.15);
xL2 = fzero(equilibrium,1-mu+0.15);
end

function limits = padded_limits(values,fraction)
values = values(isfinite(values));
assert(~isempty(values),'Cannot size axes from empty data.');
lo = min(values); hi = max(values); span = hi-lo;
if span <= 100*eps(max(1,max(abs(values))))
    span = max(0.02,0.05*max(1,abs(mean(values))));
end
padding = fraction*span;
limits = [lo-padding,hi+padding];
end

function token = sanitize_token(value)
token = lower(regexprep(string(value),'[^A-Za-z0-9]+','_'));
token = regexprep(token,'^_+|_+$','');
end

function code = mission_code(mission)
switch upper(string(mission))
    case "LUNAR_GATEWAY", code = "lg";
    case "LOW_THRUST_TRANSFER", code = "lt";
    case "GATEWAY_IMPULSE", code = "gi";
    otherwise, code = lower(string(mission));
end
end

function export_geometry_figure(fig,stem)
drawnow;
stem = string(stem);
oldUnits = fig.Units;
fig.Units = 'inches';
position = fig.Position;
fig.PaperUnits = 'inches';
fig.PaperSize = position(3:4);
fig.PaperPosition = [0 0 position(3:4)];
fig.PaperPositionMode = 'manual';
fig.Units = oldUnits;
print(fig,char(stem+".eps"),'-depsc','-painters');
exportgraphics(fig,char(stem+".png"),'Resolution',300);
end
