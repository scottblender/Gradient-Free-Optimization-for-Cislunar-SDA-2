function details = plot_reviewer2_geometry_grid(selection,figureDir,stemPrefix,saveFigures)
%PLOT_REVIEWER2_GEOMETRY_GRID Export representative 3-D result panels.
%
% Despite the historical function name, each representative realization is
% exported as its own full-size panel so it can be assembled as a LaTeX
% subfigure without shrinking several perspective axes into one MATLAB grid.
% The construction intentionally matches plot_study_definition_figures.m:
%   7.6 x 7.0 inch canvas
%   centered inner axes box [0.12 0.20 0.76 0.64]
%   perspective view(-37.5,30), axis equal/vis3d
%   8/10/10 percent x/y/z padding
%   north-outside legend centered above the restored axes box
%   Times New Roman, 12-point minimum text and 14-point axis labels
%
% Statistical claims must use aggregate mean +/- sample standard deviation.
% These plots use the saved realization nearest each group's mean objective
% only to show a physically realizable representative constellation.

if nargin < 4 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
assert(istable(selection),'selection must be a table.');
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
panels = repmat(struct(),n,1);
numObservers = nan(n,1);
families = strings(n,1);
orbitIndices = strings(n,1);
slotIndices = strings(n,1);
figureStem = strings(n,1);

% Load first so every panel for the same mission uses exactly the same
% dynamic limits. This preserves direct geometry comparability while using
% the same centered construction as the introductory figures.
missionLimits = containers.Map('KeyType','char','ValueType','any');
for mission = unique(string(selection.Mission),'stable')'
    idx = find(string(selection.Mission) == mission);
    allPoints = zeros(0,3);
    for q = 1:numel(idx)
        k = idx(q);
        panels(k) = load_geometry_panel(string(selection.RunFile(k)));
        assert(panels(k).mission == mission, ...
            'Selection mission does not match saved run mission.');
        allPoints = [allPoints;panels(k).allPoints]; %#ok<AGROW>
        numObservers(k) = height(panels(k).observers);
        families(k) = strjoin(string(panels(k).observers.orbit_family),';');
        orbitIndices(k) = strjoin(string(panels(k).observers.orbit_index),';');
        slotIndices(k) = strjoin(string(panels(k).observers.slot_index),';');
    end
    missionLimits(char(mission)) = data_limits(allPoints);
end

for k = 1:n
    mission = string(selection.Mission(k));
    panel = panels(k);
    fig = publication_figure(style.geometryFigureWidth,style.geometryFigureHeight);
    plotPosition = style.geometryPlotPosition;
    ax = axes(fig,'Units','normalized','Position',plotPosition);
    ax.PositionConstraint = 'innerposition';

    prepare_reference_axes(ax,style);
    [legendHandles,legendLabels] = render_geometry_panel(ax,panel,style);
    baseLimits = missionLimits(char(mission));
    finalize_reference_axes(ax,baseLimits,style);

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
    numObservers,families,orbitIndices,slotIndices,figureStem, ...
    'VariableNames',{'Mission','PanelKey','PanelLabel','RepresentativeObjective', ...
    'GroupMeanObjective','GroupStdObjective','RepresentativeSeed','RunFile', ...
    'NumObservers','OrbitFamilies','OrbitIndices','SlotIndices','FigureStem'});
end


function panel = load_geometry_panel(runFile)
trackingFile = string(fullfile(fileparts(runFile),'tracking_data.mat'));
assert(isfile(runFile) && isfile(trackingFile), ...
    'Missing selected run/tracking file for geometry panel.');
S = load(runFile,'runState'); T = load(trackingFile,'tracking');
r = S.runState; tracking = T.tracking;
assert(isfield(r,'observers') && istable(r.observers) && height(r.observers) >= 1, ...
    'Selected run does not contain observer solution data.');

panel.mission = string(r.settings.mission.type);
panel.truth = double(tracking.truth(:,1:3));
panel.observers = r.observers;
panel.mu = double(r.settings.mu); panel.LU = double(r.settings.LU);
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
    [departureOrbit,arrivalOrbit] = low_thrust_endpoint_orbits( ...
        tracking.truth(1,1:6),tracking.truth(end,1:6));
    panel.departureOrbit = departureOrbit(:,1:3);
    panel.arrivalOrbit = arrivalOrbit(:,1:3);
    panel.endpointOrbitPoints = [panel.departureOrbit;panel.arrivalOrbit];
else
    panel.departureOrbit = zeros(0,3); panel.arrivalOrbit = zeros(0,3);
end

moonExtent = panel.moonCenter + [ ...
    panel.moonRadius 0 0;-panel.moonRadius 0 0;0 panel.moonRadius 0; ...
    0 -panel.moonRadius 0;0 0 panel.moonRadius;0 0 -panel.moonRadius];
panel.allPoints = [panel.truth;observerPoints;panel.endpointOrbitPoints; ...
    moonExtent;panel.xL1 0 0;panel.xL2 0 0];
end


function prepare_reference_axes(ax,style)
hold(ax,'on'); box(ax,'on'); grid(ax,'off');
axis(ax,'equal');
view(ax,style.geometryAzimuth,style.geometryElevation);
ax.Projection = 'perspective';
xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
    'FontWeight','bold','LineWidth',style.axisLineWidth, ...
    'TickLabelInterpreter','tex','Layer','top');
ax.XLabel.FontName = style.fontName; ax.YLabel.FontName = style.fontName;
ax.ZLabel.FontName = style.fontName;
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
ax.ZLabel.FontSize = style.labelFontSize;
ax.XLabel.FontWeight = 'bold'; ax.YLabel.FontWeight = 'bold';
ax.ZLabel.FontWeight = 'bold';
end


function [handles,labels] = render_geometry_panel(ax,panel,style)
observerColors = lines(max(1,numel(panel.uniqueOrbitKeys)));
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

% Plot every selected phase marker, even when several observers share the
% same periodic orbit. The periodic curve itself is drawn only once above.
for j = 1:height(panel.observers)
    u = find(panel.uniqueOrbitKeys == panel.orbitKeys(j),1,'first');
    p = double(panel.observers.initial_state(j,1:3));
    plot3(ax,p(1),p(2),p(3),'o','MarkerSize',5.2, ...
        'MarkerFaceColor',observerColors(u,:),'MarkerEdgeColor','k', ...
        'LineWidth',0.7,'HandleVisibility','off');
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
else
    handles = [hTarget hObserver hMoon hL1 hL2];
    labels = ["Target trajectory","Observer orbits","Moon","L1","L2"];
end
end


function finalize_reference_axes(ax,baseLimits,style)
xlim(ax,pad_axis_limits(baseLimits(1,:),style.geometryXPadding));
ylim(ax,pad_axis_limits(baseLimits(2,:),style.geometryYPadding));
zlim(ax,pad_axis_limits(baseLimits(3,:),style.geometryZPadding));
axis(ax,'vis3d');
end


function format_case_legend(lgd,mission,style)
lgd.Box = 'on'; lgd.FontName = style.fontName; lgd.FontSize = style.fontSize;
lgd.FontWeight = 'bold'; lgd.ItemTokenSize = [16 9];
if mission == "LOW_THRUST_TRANSFER", lgd.NumColumns = 4; else, lgd.NumColumns = 5; end
end


function center_reference_legend(ax,lgd,plotPosition,style)
lgd.Units = 'normalized'; drawnow;
pos = lgd.Position;
pos(1) = 0.5-pos(3)/2;
legendBottom = plotPosition(2)+plotPosition(4)+style.geometryLegendGap;
pos(2) = min(legendBottom,0.98-pos(4));
lgd.Position = pos; lgd.AutoUpdate = 'off';
% Match the introductory figures: creating/moving a perspective legend can
% shift the axes, so restore the exact centered inner box after the legend.
ax.PositionConstraint = 'innerposition';
ax.Position = plotPosition;
drawnow;
end


function limits = data_limits(points)
assert(~isempty(points) && size(points,2) == 3,'Geometry points are empty.');
limits = zeros(3,2);
for k = 1:3
    v = points(:,k); v = v(isfinite(v));
    limits(k,:) = [min(v),max(v)];
end
end


function limits = pad_axis_limits(limits,fraction)
span = limits(2)-limits(1);
if span <= 100*eps(max(1,max(abs(limits))))
    span = max(0.02,0.05*max(1,abs(mean(limits))));
end
limits = limits+[-fraction*span,fraction*span];
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


function fig = publication_figure(widthIn,heightIn)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperPosition',[0 0 widthIn heightIn], ...
    'PaperSize',[widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
end


function export_figure(fig,figureDir,stem,saveFigures,dpi)
enforce_minimum_font_size(fig,12); drawnow;
if ~saveFigures, return; end
base = fullfile(char(figureDir),char(stem));
set(fig,'Renderer','painters','PaperPositionMode','manual');
print(fig,[base '.eps'],'-depsc2','-painters','-r600');
exportgraphics(fig,[base '.png'],'Resolution',dpi);
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
