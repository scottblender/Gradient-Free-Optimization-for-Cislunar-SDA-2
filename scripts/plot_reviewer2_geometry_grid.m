function details = plot_reviewer2_geometry_grid(selection,figureDir,stemPrefix,saveFigures)
%PLOT_REVIEWER2_GEOMETRY_GRID Export representative 3-D result panels.
%
% Historical name retained for compatibility. Final manuscript trajectories
% are exported as separate full-size panels rather than compressed MATLAB
% tiled grids. All styling/layout is completed here before the pure EPS/PNG
% exporter is called.
%
% Geometry is qualitative only. The supplied realization should be nearest
% the 20-run group mean objective; statistical conclusions use group mean
% +/- sample standard deviation. Earth is omitted. Duplicate periodic orbits
% are drawn once as solid curves while every selected phase marker is kept.
% Gateway-impulse panels retain a dashed nominal Gateway reference trajectory.

if nargin < 4 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
assert(istable(selection) && height(selection) > 0,'selection must be a nonempty table.');
required = ["Mission","PanelKey","PanelLabel","RunFile"];
assert(all(ismember(required,string(selection.Properties.VariableNames))), ...
    'Geometry selection table is missing required columns.');

style = reviewer2_paper_style();
figureDir = string(figureDir);
stemPrefix = string(stemPrefix);
if saveFigures
    assert(strlength(figureDir) > 0,'Figure directory is empty.');
    if ~isfolder(figureDir), mkdir(figureDir); end
end

n = height(selection);
panelCells = cell(n,1);
numObservers = nan(n,1);
families = strings(n,1);
orbitIndices = strings(n,1);
slotIndices = strings(n,1);
figureStem = strings(n,1);

% Load all panels first so every comparison within one mission uses the
% same dynamically determined limits.
missionLimits = containers.Map('KeyType','char','ValueType','any');
for mission = unique(string(selection.Mission),'stable')'
    idx = find(string(selection.Mission) == mission);
    allPoints = zeros(0,3);
    for q = 1:numel(idx)
        k = idx(q);
        panelCells{k} = load_geometry_panel(string(selection.RunFile(k)));
        panel = panelCells{k};
        assert(panel.mission == mission, ...
            'Selection mission does not match saved run mission.');
        allPoints = [allPoints;panel.allPoints]; %#ok<AGROW>
        numObservers(k) = height(panel.observers);
        families(k) = strjoin(string(panel.observers.orbit_family),';');
        orbitIndices(k) = strjoin(string(panel.observers.orbit_index),';');
        slotIndices(k) = strjoin(string(panel.observers.slot_index),';');
    end
    missionLimits(char(mission)) = common_geometry_limits(allPoints,style);
end

for k = 1:n
    mission = string(selection.Mission(k));
    panel = panelCells{k};
    fig = publication_figure(style.geometryFigureWidth,style.geometryFigureHeight,style);
    plotPosition = style.geometryPlotPosition;
    ax = axes(fig,'Units','normalized','Position',plotPosition);
    ax.PositionConstraint = 'innerposition';

    prepare_reference_axes(ax,style,mission);
    [legendHandles,legendLabels] = render_geometry_panel(ax,panel,style);
    limits = missionLimits(char(mission));
    xlim(ax,limits(1,:)); ylim(ax,limits(2,:)); zlim(ax,limits(3,:));
    axis(ax,'vis3d');

    % Final trajectory tick selection happens during figure generation, not
    % during export. This removes a projected center-zero label only for the
    % symmetric three-tick case that caused 0 / 0.05 overlap.
    simplify_geometry_ticks(ax);

    legendLabels = compact_geometry_labels(legendLabels);
    legendHandle = legend(ax,legendHandles,cellstr(legendLabels), ...
        'Location','northoutside','Orientation','horizontal');
    format_case_legend(legendHandle,panel.mission,style);
    center_reference_legend(ax,legendHandle,plotPosition,style);

    stem = stemPrefix + "_" + mission_code(mission) + "_" + ...
        sanitize_key(string(selection.PanelKey(k)));
    figureStem(k) = stem;
    export_figure(fig,figureDir,stem,saveFigures,style.exportDpi);
end

representativeObjective = optional_numeric(selection,'RepresentativeObjective');
groupMeanObjective = optional_numeric(selection,'GroupMeanObjective');
groupStdObjective = optional_numeric(selection,'GroupStdObjective');
representativeSeed = optional_numeric(selection,'RepresentativeSeed');
if all(isnan(representativeSeed)) && ismember('Seed',selection.Properties.VariableNames)
    representativeSeed = double(selection.Seed);
end

details = table(string(selection.Mission),string(selection.PanelKey), ...
    string(selection.PanelLabel),representativeObjective,groupMeanObjective, ...
    groupStdObjective,representativeSeed,string(selection.RunFile), ...
    numObservers,families,orbitIndices,slotIndices,figureStem,figureStem, ...
    'VariableNames',{'Mission','PanelKey','PanelLabel','RepresentativeObjective', ...
    'GroupMeanObjective','GroupStdObjective','RepresentativeSeed','RunFile', ...
    'NumObservers','OrbitFamilies','OrbitIndices','SlotIndices', ...
    'FigureStem','GridFigureStem'});
end


function panel = load_geometry_panel(runFile)
trackingFile = string(fullfile(fileparts(runFile),'tracking_data.mat'));
assert(isfile(runFile) && isfile(trackingFile), ...
    'Missing selected run/tracking file for geometry panel.');
S = load(runFile,'runState');
T = load(trackingFile,'tracking');
r = S.runState;
tracking = T.tracking;
assert(isfield(r,'observers') && istable(r.observers) && height(r.observers) >= 1, ...
    'Selected run does not contain observer solution data.');

panel.mission = string(r.settings.mission.type);
panel.truth = double(tracking.truth(:,1:3));
panel.observers = r.observers;
panel.mu = double(r.settings.mu);
panel.LU = double(r.settings.LU);
panel.targetColor = reviewer2_target_color(panel.mission);
panel.moonCenter = [1-panel.mu,0,0];
panel.moonRadius = 1737.1/panel.LU;
[panel.xL1,panel.xL2] = collinear_lagrange_points(panel.mu);

family = string(panel.observers.orbit_family);
orbitIndex = string(panel.observers.orbit_index);
panel.orbitKeys = family + "_" + orbitIndex;
panel.uniqueOrbitKeys = unique(panel.orbitKeys,'stable');
panel.orbitTrajectories = cell(numel(panel.uniqueOrbitKeys),1);
opts = odeset('RelTol',1e-11,'AbsTol',1e-12);
observerPoints = zeros(0,3);
for u = 1:numel(panel.uniqueOrbitKeys)
    member = find(panel.orbitKeys == panel.uniqueOrbitKeys(u),1,'first');
    period = double(panel.observers.period_TU(member));
    validateattributes(period,{'numeric'},{'scalar','real','finite','positive'});
    tPlot = linspace(0,period,360);
    initialState = double(panel.observers.initial_state(member,:))';
    [~,state] = ode45(@(t,s) cr3bp_dynamics(t,s,panel.mu),tPlot,initialState,opts);
    panel.orbitTrajectories{u} = state(:,1:3);
    observerPoints = [observerPoints;state(:,1:3)]; %#ok<AGROW>
end

panel.endpointOrbitPoints = zeros(0,3);
if panel.mission == "LOW_THRUST_TRANSFER"
    assert(size(tracking.truth,2) >= 6, ...
        'Low-thrust geometry requires six-component truth states.');
    [departureOrbit,arrivalOrbit] = low_thrust_endpoint_orbits( ...
        tracking.truth(1,1:6),tracking.truth(end,1:6));
    panel.departureOrbit = departureOrbit(:,1:3);
    panel.arrivalOrbit = arrivalOrbit(:,1:3);
    panel.endpointOrbitPoints = [panel.departureOrbit;panel.arrivalOrbit];
else
    panel.departureOrbit = zeros(0,3);
    panel.arrivalOrbit = zeros(0,3);
end

panel.nominalGateway = zeros(0,3);
if panel.mission == "GATEWAY_IMPULSE"
    panel.nominalGateway = nominal_gateway_reference(panel.mu);
end

moonExtent = panel.moonCenter + [ ...
    panel.moonRadius 0 0;-panel.moonRadius 0 0; ...
    0 panel.moonRadius 0;0 -panel.moonRadius 0; ...
    0 0 panel.moonRadius;0 0 -panel.moonRadius];
panel.allPoints = [panel.truth;observerPoints;panel.endpointOrbitPoints; ...
    panel.nominalGateway;moonExtent;panel.xL1 0 0;panel.xL2 0 0];
end


function prepare_reference_axes(ax,style,mission)
hold(ax,'on'); box(ax,'off'); grid(ax,'off');
axis(ax,'equal');
missionKey = char(upper(string(mission)));
assert(isfield(style.maneuverViews,missionKey), ...
    'No maneuver camera view configured for %s.',missionKey);
assert(isfield(style.maneuverProjections,missionKey), ...
    'No maneuver camera projection configured for %s.',missionKey);
viewAngles = style.maneuverViews.(missionKey);
view(ax,viewAngles(1),viewAngles(2));
ax.Projection = style.maneuverProjections.(missionKey);
xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
    'FontWeight',style.fontWeight,'LineWidth',style.axisLineWidth, ...
    'TickLabelInterpreter','tex','Layer','top', ...
    'Box','off','XGrid','off','YGrid','off','ZGrid','off');
ax.XLabel.FontName = style.fontName; ax.YLabel.FontName = style.fontName;
ax.ZLabel.FontName = style.fontName;
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
ax.ZLabel.FontSize = style.labelFontSize;
ax.XLabel.FontWeight = style.fontWeight; ax.YLabel.FontWeight = style.fontWeight;
ax.ZLabel.FontWeight = style.fontWeight;
end


function [handles,labels] = render_geometry_panel(ax,panel,~)
observerColors = lines(max(1,numel(panel.uniqueOrbitKeys)));

hNominal = gobjects(0);
if panel.mission == "GATEWAY_IMPULSE"
    gatewayColor = reviewer2_target_color("LUNAR_GATEWAY");
    nominalAlpha = 0.45;
    nominalColor = nominalAlpha*gatewayColor+(1-nominalAlpha)*[1 1 1];
    hNominal = plot3(ax,panel.nominalGateway(:,1),panel.nominalGateway(:,2), ...
        panel.nominalGateway(:,3),'--','Color',nominalColor,'LineWidth',2.2, ...
        'DisplayName','Nominal Gateway');
end

hTarget = plot3(ax,panel.truth(:,1),panel.truth(:,2),panel.truth(:,3),'-', ...
    'Color',panel.targetColor,'LineWidth',2.8,'DisplayName','Target trajectory');

hObserver = gobjects(1,1);
for u = 1:numel(panel.uniqueOrbitKeys)
    state = panel.orbitTrajectories{u};
    h = plot3(ax,state(:,1),state(:,2),state(:,3),'-', ...
        'Color',observerColors(u,:),'LineWidth',1.45,'HandleVisibility','off');
    if u == 1, hObserver = h; end
end
set(hObserver,'HandleVisibility','on','DisplayName','Observer orbits');

for j = 1:height(panel.observers)
    u = find(panel.uniqueOrbitKeys == panel.orbitKeys(j),1,'first');
    p = double(panel.observers.initial_state(j,1:3));
    plot3(ax,p(1),p(2),p(3),'o','MarkerSize',5.5, ...
        'MarkerFaceColor',observerColors(u,:),'MarkerEdgeColor','k', ...
        'LineWidth',0.75,'HandleVisibility','off');
end

hEndpoint = gobjects(0); hStart = gobjects(0); hEnd = gobjects(0);
if panel.mission == "LOW_THRUST_TRANSFER"
    endpointColor = [0.70 0.70 0.70];
    hEndpoint = plot3(ax,panel.departureOrbit(:,1),panel.departureOrbit(:,2), ...
        panel.departureOrbit(:,3),'-','Color',endpointColor,'LineWidth',1.0, ...
        'DisplayName','Endpoint orbits');
    plot3(ax,panel.arrivalOrbit(:,1),panel.arrivalOrbit(:,2), ...
        panel.arrivalOrbit(:,3),'-','Color',endpointColor,'LineWidth',1.0, ...
        'HandleVisibility','off');
    hStart = plot3(ax,panel.truth(1,1),panel.truth(1,2),panel.truth(1,3),'o', ...
        'MarkerSize',9,'MarkerFaceColor',reviewer2_target_color("LUNAR_GATEWAY"), ...
        'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','Start');
    hEnd = plot3(ax,panel.truth(end,1),panel.truth(end,2),panel.truth(end,3),'s', ...
        'MarkerSize',9,'MarkerFaceColor',panel.targetColor, ...
        'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','End');
end

[sx,sy,sz] = sphere(30);
hMoon = surf(ax,panel.moonCenter(1)+panel.moonRadius*sx, ...
    panel.moonCenter(2)+panel.moonRadius*sy, ...
    panel.moonCenter(3)+panel.moonRadius*sz, ...
    'FaceColor',[0.72 0.72 0.72],'EdgeColor','none', ...
    'FaceLighting','gouraud','DisplayName','Moon');
hL1 = plot3(ax,panel.xL1,0,0,'^','MarkerSize',9, ...
    'MarkerFaceColor',[0.80 0.80 0.80],'MarkerEdgeColor','k', ...
    'LineWidth',1.2,'DisplayName','L1');
hL2 = plot3(ax,panel.xL2,0,0,'v','MarkerSize',9, ...
    'MarkerFaceColor',[0.80 0.80 0.80],'MarkerEdgeColor','k', ...
    'LineWidth',1.2,'DisplayName','L2');
camlight(ax,'headlight'); material(ax,'dull');

if panel.mission == "LOW_THRUST_TRANSFER"
    handles = [hEndpoint hTarget hObserver hStart hEnd hMoon hL1 hL2];
    labels = ["Endpoint orbits","Target trajectory","Observer orbits", ...
        "Start","End","Moon","L1","L2"];
elseif panel.mission == "GATEWAY_IMPULSE"
    handles = [hNominal hTarget hObserver hMoon hL1 hL2];
    labels = ["Nominal Gateway","Target trajectory","Observer orbits", ...
        "Moon","L1","L2"];
else
    handles = [hTarget hObserver hMoon hL1 hL2];
    labels = ["Target trajectory","Observer orbits","Moon","L1","L2"];
end
end


function labels = compact_geometry_labels(labels)
labels = string(labels);
labels(labels=="Endpoint orbits") = "Endpoints";
labels(labels=="Target trajectory") = "Target";
labels(labels=="Observer orbits") = "Obs. orbits";
labels(labels=="Nominal Gateway") = "Nominal LG";
labels(labels=="Post-impulse") = "GI traj.";
end


function format_case_legend(lgd,mission,style)
lgd.Box = 'off';
lgd.FontName = style.fontName;
lgd.FontSize = style.fontSize;
lgd.FontWeight = style.fontWeight;
lgd.ItemTokenSize = [16 9];
if mission == "LOW_THRUST_TRANSFER"
    lgd.NumColumns = 4;
elseif mission == "GATEWAY_IMPULSE"
    lgd.NumColumns = 3;
else
    lgd.NumColumns = 5;
end
lgd.NumColumns = min(lgd.NumColumns,style.legendMaxColumns);
end


function center_reference_legend(ax,lgd,plotPosition,style)
% Let MATLAB establish true northoutside size first, then lower it slightly
% and freeze that final generated position before export.
lgd.Location = 'northoutside';
lgd.Units = 'normalized';
drawnow;
northPosition = lgd.Position;
lgd.Location = 'none';

ax.PositionConstraint = 'innerposition';
ax.Position = plotPosition;
drawnow;
pos = lgd.Position;
pos(1) = max(0.002,min(0.5-pos(3)/2,0.998-pos(3)));
minimumBottom = plotPosition(2)+plotPosition(4)+style.legendMinimumGap;
desiredBottom = northPosition(2)+style.legendNorthOutsideYOffset;
pos(2) = min(max(desiredBottom,minimumBottom),0.99-pos(4));
lgd.Position = pos;
lgd.AutoUpdate = 'off';
ax.Position = plotPosition;
drawnow;
end


function simplify_geometry_ticks(ax)
for axisName = ["X","Y","Z"]
    tickProperty = axisName+"Tick";
    labelProperty = axisName+"TickLabel";
    ticks = double(ax.(tickProperty));
    if numel(ticks)~=3 || any(~isfinite(ticks)), continue; end
    tolerance = 100*eps(max(1,max(abs(ticks))));
    if abs(ticks(2))<=tolerance && abs(ticks(1)+ticks(3))<=tolerance
        labels = ax.(labelProperty);
        ax.(tickProperty) = ticks([1 3]);
        if ~isempty(labels) && size(labels,1)==3
            if iscell(labels) || isstring(labels)
                ax.(labelProperty) = labels([1 3]);
            elseif ischar(labels)
                ax.(labelProperty) = labels([1 3],:);
            end
        end
    end
end
end


function limits = common_geometry_limits(points,style)
assert(~isempty(points) && size(points,2) == 3,'Geometry points are empty.');
padding = [style.geometryXPadding style.geometryYPadding style.geometryZPadding];
limits = zeros(3,2);
for k = 1:3
    v = points(:,k); v = v(isfinite(v));
    assert(~isempty(v),'Geometry coordinate data are empty.');
    lo = min(v); hi = max(v); span = hi-lo;
    if span <= 100*eps(max(1,max(abs(v))))
        span = max(0.02,0.05*max(1,abs(mean(v))));
    end
    limits(k,:) = [lo-padding(k)*span,hi+padding(k)*span];
end
end


function values = optional_numeric(T,name)
values = nan(height(T),1);
if ismember(name,T.Properties.VariableNames), values = double(T.(name)); end
end


function [departureOrbit,arrivalOrbit] = low_thrust_endpoint_orbits(startState,endState)
persistent cachedStart cachedEnd cachedDeparture cachedArrival
startState = double(startState(:).'); endState = double(endState(:).');
if ~isempty(cachedStart) && isequal(size(cachedStart),size(startState)) && ...
        max(abs(cachedStart-startState)) < 1e-12 && ...
        max(abs(cachedEnd-endState)) < 1e-12
    departureOrbit = cachedDeparture; arrivalOrbit = cachedArrival; return;
end
paths = setup_project(); catalog = load(paths.catalog,'T');
departureOrbit = find_reference_orbit_for_state(catalog.T,startState);
arrivalOrbit = find_reference_orbit_for_state(catalog.T,endState);
cachedStart = startState; cachedEnd = endState;
cachedDeparture = departureOrbit; cachedArrival = arrivalOrbit;
end


function orbitState = find_reference_orbit_for_state(T,targetState)
assert(istable(T) && ismember('state',T.Properties.VariableNames), ...
    'Observer catalog must contain the state trajectory column.');
targetState = double(targetState(:).'); bestError = inf; bestOrbit = [];
for k = 1:height(T)
    state = T.state{k};
    if isempty(state) || size(state,2) < 6, continue; end
    state6 = double(state(:,1:6)); state6 = state6(all(isfinite(state6),2),:);
    if isempty(state6), continue; end
    thisError = min(vecnorm(state6-targetState,2,2));
    if thisError < bestError, bestError = thisError; bestOrbit = state6; end
end
assert(~isempty(bestOrbit) && isfinite(bestError), ...
    'Could not identify a low-thrust endpoint reference orbit.');
assert(bestError < 2.5e-2,'Low-thrust endpoint reference-orbit mismatch: %.6e.',bestError);
orbitState = bestOrbit;
end


function nominalState = nominal_gateway_reference(mu)
persistent cachedMu cachedState
mu = double(mu);
if ~isempty(cachedMu) && abs(cachedMu-mu) <= 10*eps(max(1,abs(mu)))
    nominalState = cachedState;
    return;
end
cfg = target_case_config("GATEWAY_IMPULSE");
opts = odeset('RelTol',1e-13,'AbsTol',1e-13);
[tImpulse,~,info] = build_target_truth(cfg,table(),{}, {}, {},mu,opts);
[~,sNominal] = ode45(@(t,s) cr3bp_dynamics(t,s,mu), ...
    tImpulse,info.statePreImpulse,opts);
nominalState = double(sNominal(:,1:3));
cachedMu = mu;
cachedState = nominalState;
end


function [xL1,xL2] = collinear_lagrange_points(mu)
equilibrium = @(x) x -(1-mu)*(x+mu)./abs(x+mu).^3 ...
    -mu*(x-1+mu)./abs(x-1+mu).^3;
xL1 = fzero(equilibrium,1-mu-0.15);
xL2 = fzero(equilibrium,1-mu+0.15);
end


function code = mission_code(mission)
switch upper(string(mission))
    case "LUNAR_GATEWAY", code = "lg";
    case "LOW_THRUST_TRANSFER", code = "lt";
    case "GATEWAY_IMPULSE", code = "gi";
    otherwise, code = lower(string(mission));
end
end


function key = sanitize_key(value)
key = lower(regexprep(string(value),'[^a-zA-Z0-9]+','_'));
key = strip(key,'_');
end


function fig = publication_figure(widthIn,heightIn,style)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperPosition',[0 0 widthIn heightIn], ...
    'PaperSize',[widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
% Apply manuscript typography defaults before any axes/text/legend is made.
set(fig,'DefaultAxesFontName',style.fontName, ...
    'DefaultAxesFontSize',style.fontSize, ...
    'DefaultAxesFontWeight',style.fontWeight, ...
    'DefaultTextFontName',style.fontName, ...
    'DefaultTextFontSize',style.fontSize, ...
    'DefaultTextFontWeight',style.fontWeight);
end


function export_figure(fig,figureDir,stem,saveFigures,~)
if ~saveFigures, return; end
drawnow;
export_manuscript_figure(fig,fullfile(char(figureDir),char(stem)));
close(fig);
end


function enforce_minimum_font_size(fig,minFontSize)
objects = findall(fig,'-property','FontSize');
for k = 1:numel(objects)
    try
        if objects(k).FontSize < minFontSize, objects(k).FontSize = minFontSize; end
    catch
    end
end
end
