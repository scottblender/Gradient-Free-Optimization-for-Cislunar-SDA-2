function details = plot_reviewer2_constellation_geometry(selection,figureDir,stemPrefix,saveFigures)
%PLOT_REVIEWER2_CONSTELLATION_GEOMETRY Plot selected observer constellations.
%
% Each row in selection is one manuscript panel. The selected observer
% trajectories are propagated for one period of each observer orbit and are
% shown with the saved target truth trajectory. This exposes constellation
% geometry directly, rather than plotting only target/EKF estimation error.
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
truth = tracking.truth(:,1:3);
observers = runState.observers;
nObs = height(observers);

fig = figure('Color','w','Units','inches','Position',[1 1 7.6 7.0], ...
    'PaperUnits','inches','PaperSize',[7.6 7.0], ...
    'PaperPosition',[0 0 7.6 7.0],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
plotPosition = [0.12 0.20 0.76 0.64];
ax = axes(fig,'Units','normalized','Position',plotPosition);
ax.PositionConstraint = 'innerposition';
hold(ax,'on'); box(ax,'on'); axis(ax,'equal');

hTarget = plot3(ax,truth(:,1),truth(:,2),truth(:,3),'k--', ...
    'LineWidth',2.5,'DisplayName','Target trajectory');

observerColors = lines(max(nObs,1));
allObserverPoints = zeros(0,3);
hObserver = gobjects(1,1);
opts = odeset('RelTol',1e-11,'AbsTol',1e-12);
for j = 1:nObs
    period = double(observers.period_TU(j));
    validateattributes(period,{'numeric'},{'scalar','real','finite','positive'});
    tPlot = linspace(0,period,300);
    initialState = observers.initial_state(j,:)';
    [~,state] = ode45(@(t,s) cr3bp_dynamics(t,s,mu),tPlot,initialState,opts);
    allObserverPoints = [allObserverPoints;state(:,1:3)]; %#ok<AGROW>
    h = plot3(ax,state(:,1),state(:,2),state(:,3),'-', ...
        'Color',observerColors(j,:),'LineWidth',1.65, ...
        'HandleVisibility','off');
    plot3(ax,state(1,1),state(1,2),state(1,3),'o', ...
        'MarkerSize',5.5,'MarkerFaceColor',observerColors(j,:), ...
        'MarkerEdgeColor','k','LineWidth',0.7,'HandleVisibility','off');
    if j == 1, hObserver = h; end
end
set(hObserver,'HandleVisibility','on','DisplayName','Observer orbits');

earthCenter = [-mu,0,0];
moonCenter = [1-mu,0,0];
earthRadius = 6378.1366/LU;
moonRadius = 1737.1/LU;
[sx,sy,sz] = sphere(30);
hEarth = surf(ax,earthCenter(1)+earthRadius*sx, ...
    earthCenter(2)+earthRadius*sy,earthCenter(3)+earthRadius*sz, ...
    'FaceColor',[0.62 0.68 0.74],'EdgeColor','none', ...
    'FaceLighting','gouraud','DisplayName','Earth');
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

allPoints = [truth;allObserverPoints;earthCenter;moonCenter;xL1 0 0;xL2 0 0];
xlim(ax,padded_limits(allPoints(:,1),0.06));
ylim(ax,padded_limits(allPoints(:,2),0.08));
zlim(ax,padded_limits(allPoints(:,3),0.08));
axis(ax,'vis3d'); ax.Projection = 'perspective'; view(ax,-37.5,30); grid(ax,'off');

xlabel(ax,'x (LU)','FontWeight','bold');
ylabel(ax,'y (LU)','FontWeight','bold');
zlabel(ax,'z (LU)','FontWeight','bold');
set(ax,'FontName','Times New Roman','FontSize',12,'FontWeight','bold', ...
    'LineWidth',1.2,'Layer','top');
ax.XLabel.FontSize = 14; ax.YLabel.FontSize = 14; ax.ZLabel.FontSize = 14;

lgd = legend(ax,[hTarget hObserver hEarth hMoon hL1 hL2], ...
    {'Target trajectory','Observer orbits','Earth','Moon','L1','L2'}, ...
    'Orientation','horizontal','NumColumns',3,'Box','on');
lgd.FontName = 'Times New Roman'; lgd.FontSize = 12; lgd.FontWeight = 'bold';
lgd.ItemTokenSize = [18 10]; lgd.Units = 'normalized';
drawnow;
legendPosition = lgd.Position;
legendPosition(1) = 0.5-legendPosition(3)/2;
legendBottom = plotPosition(2)+plotPosition(4)+0.012;
legendPosition(2) = min(legendBottom,0.98-legendPosition(4));
lgd.Position = legendPosition; lgd.AutoUpdate = 'off';
ax.Position = plotPosition;
drawnow;
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
