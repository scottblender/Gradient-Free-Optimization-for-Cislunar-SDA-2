function details = plot_reviewer2_geometry_grid(selection,figureDir,stemPrefix,saveFigures)
%PLOT_REVIEWER2_GEOMETRY_GRID Fixed-size journal constellation geometry grids.
%
% Geometry panels are qualitative representatives only. Statistical claims
% must use the 20-run aggregate mean +/- sample standard deviation. The
% selection table is expected to contain a representative realization near
% each aggregate mean, not the lowest-cost seed.
%
% Every mission grid is exported on a fixed 6.5 x 6.5 inch canvas. Axes use
% manually assigned OuterPosition boxes so tick labels and x/y/z labels stay
% inside the page. A common camera and common limits are used within a
% mission so differences between panels are geometric rather than plotting
% artifacts. Earth is intentionally omitted. All orbit lines are solid.

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
numObservers = nan(n,1);
families = strings(n,1);
orbitIndices = strings(n,1);
slotIndices = strings(n,1);
figureStem = strings(n,1);

for mission = unique(string(selection.Mission),'stable')'
    idx = find(string(selection.Mission) == mission);
    panels = repmat(struct(),numel(idx),1);
    allPoints = zeros(0,3);

    for q = 1:numel(idx)
        k = idx(q);
        panels(q) = load_geometry_panel(string(selection.RunFile(k)));
        assert(panels(q).mission == mission, ...
            'Selection mission does not match saved run mission.');
        allPoints = [allPoints;panels(q).allPoints]; %#ok<AGROW>
        numObservers(k) = height(panels(q).observers);
        families(k) = strjoin(string(panels(q).observers.orbit_family),';');
        orbitIndices(k) = strjoin(string(panels(q).observers.orbit_index),';');
        slotIndices(k) = strjoin(string(panels(q).observers.slot_index),';');
    end

    limits = common_geometry_limits(allPoints);
    nPanels = numel(idx);
    if nPanels <= 2
        nRows = 1; nColumns = nPanels;
    else
        nColumns = 2; nRows = ceil(nPanels/2);
    end

    fig = paper_figure(style.geometryFigureWidth,style.geometryFigureHeight);
    positions = panel_outer_positions(nRows,nColumns,nPanels);
    legendHandles = gobjects(0); legendLabels = strings(0,1);

    for q = 1:nPanels
        ax = axes(fig,'Units','normalized','OuterPosition',positions(q,:), ...
            'PositionConstraint','outerposition');
        [handles,labels] = render_geometry_panel(ax,panels(q),limits,style);
        title(ax,sprintf('(%c) %s',char('a'+q-1),string(selection.PanelLabel(idx(q)))), ...
            'FontName',style.fontName,'FontSize',style.fontSize, ...
            'FontWeight','bold','Interpreter','none');
        if q == 1
            legendHandles = handles; legendLabels = labels;
        end
    end

    lgd = legend(legendHandles,cellstr(legendLabels), ...
        'Orientation','horizontal','NumColumns',min(4,numel(legendLabels)), ...
        'Box','on');
    lgd.Units = 'normalized';
    lgd.FontName = style.fontName; lgd.FontSize = style.fontSize;
    lgd.FontWeight = 'bold'; lgd.ItemTokenSize = [16 9];
    drawnow;
    legendHeight = 0.055;
    if numel(legendLabels) > 5, legendHeight = 0.09; end
    lgd.Position = [0.08,0.965-legendHeight,0.84,legendHeight];

    % Reapply axes outer boxes after legend creation. MATLAB legends can
    % otherwise shrink 3-D axes even when the intended export size is fixed.
    for q = 1:nPanels
        ax = fig.Children;
    end
    axesList = findobj(fig,'Type','axes');
    axesList = flipud(axesList(:));
    for q = 1:min(nPanels,numel(axesList))
        axesList(q).OuterPosition = positions(q,:);
    end
    drawnow;

    stem = stemPrefix + "_" + mission_code(mission) + "_grid";
    figureStem(idx) = stem;
    export_figure(fig,figureDir,stem,saveFigures,style.exportDpi);
end

representativeObjective = nan(n,1);
groupMeanObjective = nan(n,1); groupStdObjective = nan(n,1);
representativeSeed = nan(n,1);
if ismember('RepresentativeObjective',selection.Properties.VariableNames)
    representativeObjective = double(selection.RepresentativeObjective);
end
if ismember('GroupMeanObjective',selection.Properties.VariableNames)
    groupMeanObjective = double(selection.GroupMeanObjective);
end
if ismember('GroupStdObjective',selection.Properties.VariableNames)
    groupStdObjective = double(selection.GroupStdObjective);
end
if ismember('RepresentativeSeed',selection.Properties.VariableNames)
    representativeSeed = double(selection.RepresentativeSeed);
elseif ismember('Seed',selection.Properties.VariableNames)
    representativeSeed = double(selection.Seed);
end

details = table(string(selection.Mission),string(selection.PanelKey), ...
    string(selection.PanelLabel),representativeObjective,groupMeanObjective, ...
    groupStdObjective,representativeSeed,string(selection.RunFile), ...
    numObservers,families,orbitIndices,slotIndices,figureStem, ...
    'VariableNames',{'Mission','PanelKey','PanelLabel','RepresentativeObjective', ...
    'GroupMeanObjective','GroupStdObjective','RepresentativeSeed','RunFile', ...
    'NumObservers','OrbitFamilies','OrbitIndices','SlotIndices','GridFigureStem'});
end


function panel = load_geometry_panel(runFile)
trackingFile = string(fullfile(fileparts(runFile),'tracking_data.mat'));
assert(isfile(runFile) && isfile(trackingFile), ...
    'Missing selected run/tracking file for geometry panel.');
S = load(runFile,'runState'); T = load(trackingFile,'tracking');
r = S.runState; tracking = T.tracking;
assert(isfield(r,'observers') && istable(r.observers) && height(r.observers) >= 1, ...
    'Selected run does not contain observer solution data.');

panel.runState = r; panel.tracking = tracking;
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


function [handles,labels] = render_geometry_panel(ax,panel,limits,style)
hold(ax,'on'); box(ax,'on'); grid(ax,'off');
axis(ax,'vis3d'); daspect(ax,[1 1 1]);
observerColors = lines(max(1,numel(panel.uniqueOrbitKeys)));

hTarget = plot3(ax,panel.truth(:,1),panel.truth(:,2),panel.truth(:,3),'-', ...
    'Color',panel.targetColor,'LineWidth',2.25,'DisplayName','Target trajectory');

hObserver = gobjects(1,1);
for u = 1:numel(panel.uniqueOrbitKeys)
    state = panel.orbitTrajectories{u};
    h = plot3(ax,state(:,1),state(:,2),state(:,3), ...
        'LineStyle','-','Color',observerColors(u,:), ...
        'LineWidth',1.35,'HandleVisibility','off');
    if u == 1, hObserver = h; end
end
set(hObserver,'HandleVisibility','on','DisplayName','Observer orbits');

for j = 1:height(panel.observers)
    u = find(panel.uniqueOrbitKeys == panel.orbitKeys(j),1,'first');
    p = double(panel.observers.initial_state(j,1:3));
    plot3(ax,p(1),p(2),p(3),'o','MarkerSize',4.8, ...
        'MarkerFaceColor',observerColors(u,:),'MarkerEdgeColor','k', ...
        'LineWidth',0.55,'HandleVisibility','off');
end

hEndpoint = gobjects(0); hStart = gobjects(0); hEnd = gobjects(0);
if panel.mission == "LOW_THRUST_TRANSFER"
    endpointColor = [0.66 0.66 0.66];
    hEndpoint = plot3(ax,panel.departureOrbit(:,1),panel.departureOrbit(:,2), ...
        panel.departureOrbit(:,3),'-','Color',endpointColor,'LineWidth',0.9, ...
        'DisplayName','Endpoint orbits');
    plot3(ax,panel.arrivalOrbit(:,1),panel.arrivalOrbit(:,2), ...
        panel.arrivalOrbit(:,3),'-','Color',endpointColor,'LineWidth',0.9, ...
        'HandleVisibility','off');
    hStart = plot3(ax,panel.truth(1,1),panel.truth(1,2),panel.truth(1,3),'o', ...
        'MarkerSize',5.5,'MarkerFaceColor',reviewer2_target_color("LUNAR_GATEWAY"), ...
        'MarkerEdgeColor','k','LineWidth',0.7,'DisplayName','Start');
    hEnd = plot3(ax,panel.truth(end,1),panel.truth(end,2),panel.truth(end,3),'s', ...
        'MarkerSize',5.5,'MarkerFaceColor',panel.targetColor, ...
        'MarkerEdgeColor','k','LineWidth',0.7,'DisplayName','End');
end

[sx,sy,sz] = sphere(20);
hMoon = surf(ax,panel.moonCenter(1)+panel.moonRadius*sx, ...
    panel.moonCenter(2)+panel.moonRadius*sy, ...
    panel.moonCenter(3)+panel.moonRadius*sz, ...
    'FaceColor',[0.72 0.72 0.72],'EdgeColor','none', ...
    'FaceLighting','gouraud','DisplayName','Moon');
hL1 = plot3(ax,panel.xL1,0,0,'^','MarkerSize',6.5, ...
    'MarkerFaceColor',[0.82 0.82 0.82],'MarkerEdgeColor','k', ...
    'LineWidth',0.8,'DisplayName','L1');
hL2 = plot3(ax,panel.xL2,0,0,'v','MarkerSize',6.5, ...
    'MarkerFaceColor',[0.82 0.82 0.82],'MarkerEdgeColor','k', ...
    'LineWidth',0.8,'DisplayName','L2');
camlight(ax,'headlight'); material(ax,'dull');

xlim(ax,limits(1,:)); ylim(ax,limits(2,:)); zlim(ax,limits(3,:));
ax.Projection = 'perspective';
view(ax,style.geometryAzimuth,style.geometryElevation);
xlabel(ax,'x (LU)','FontWeight','bold');
ylabel(ax,'y (LU)','FontWeight','bold');
zlabel(ax,'z (LU)','FontWeight','bold');
set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
    'FontWeight','bold','LineWidth',style.axisLineWidth,'TickDir','out', ...
    'PositionConstraint','outerposition');
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
ax.ZLabel.FontSize = style.labelFontSize;

if panel.mission == "LOW_THRUST_TRANSFER"
    handles = [hEndpoint hTarget hObserver hStart hEnd hMoon hL1 hL2];
    labels = ["Endpoint orbits","Target trajectory","Observer orbits", ...
        "Start","End","Moon","L1","L2"];
else
    handles = [hTarget hObserver hMoon hL1 hL2];
    labels = ["Target trajectory","Observer orbits","Moon","L1","L2"];
end
end


function positions = panel_outer_positions(nRows,nColumns,nPanels)
% Manual outer boxes keep 12/14-point 3-D labels inside a 6.5-inch export.
left = 0.055; right = 0.035; bottom = 0.055; top = 0.165;
gapX = 0.015; gapY = 0.025;
width = (1-left-right-(nColumns-1)*gapX)/nColumns;
height = (1-bottom-top-(nRows-1)*gapY)/nRows;
positions = nan(nPanels,4);
for q = 1:nPanels
    row = floor((q-1)/nColumns);
    col = mod(q-1,nColumns);
    x = left + col*(width+gapX);
    y = 1-top-(row+1)*height-row*gapY;
    positions(q,:) = [x y width height];
end
end


function limits = common_geometry_limits(points)
assert(~isempty(points) && size(points,2) == 3,'Geometry points are empty.');
limits = zeros(3,2);
for k = 1:3
    v = points(:,k); v = v(isfinite(v));
    lo = min(v); hi = max(v); span = hi-lo;
    if span <= 100*eps(max(1,max(abs(v))))
        span = max(0.02,0.05*max(1,abs(mean(v))));
    end
    padding = 0.065*span;
    limits(k,:) = [lo-padding,hi+padding];
end
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


function fig = paper_figure(widthIn,heightIn)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
end


function export_figure(fig,figureDir,stem,saveFigures,dpi)
drawnow;
if ~saveFigures, return; end
assert(strlength(string(figureDir)) > 0,'Figure directory is empty.');
base = fullfile(char(figureDir),char(stem));
print(fig,[base '.eps'],'-depsc','-painters');
exportgraphics(fig,[base '.png'],'Resolution',dpi);
close(fig);
end
