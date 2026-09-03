function figureFiles = plot_reviewer2_best_trajectories(analysisDir,saveFigures)
%PLOT_REVIEWER2_BEST_TRAJECTORIES Regenerate centered best-run trajectories.
%
% This plotting-only helper uses the processed Reviewer 2 pilot outputs and
% does not rerun any optimization. It matches the study-definition CR3BP
% camera (perspective, view(-37.5,30)), emphasizes the EKF estimate, and
% uses a centered 3-D plotting region with explicit margins so the projected
% x/y/z axis labels remain inside the export canvas.
%
% Usage:
%   plot_reviewer2_best_trajectories
%   plot_reviewer2_best_trajectories(analysisDir)
%   plot_reviewer2_best_trajectories(analysisDir,false)

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);

paths = setup_project();
if nargin < 1 || isempty(analysisDir)
    pilotRoot = fullfile(paths.results,'COMPARISON_PILOT_1200');
    analysisDir = newest_analysis_directory(pilotRoot);
else
    analysisDir = char(string(analysisDir));
end
assert(isfolder(analysisDir),'Analysis directory does not exist: %s',analysisDir);

bestRunFile = fullfile(analysisDir,'pilot_best_observed_runs.csv');
assert(isfile(bestRunFile),'Missing best-run table: %s',bestRunFile);
bestRuns = readtable(bestRunFile,'TextType','string', ...
    'VariableNamingRule','preserve');
required = ["Mission","Optimizer","Seed","OptimizationRunFile","TrackingDataFile"];
assert(all(ismember(required,string(bestRuns.Properties.VariableNames))), ...
    'Best-run table is missing required trajectory fields.');

figureDir = fullfile(analysisDir,'paper_preview');
if saveFigures && ~isfolder(figureDir), mkdir(figureDir); end
figureFiles = strings(height(bestRuns),2);

for k = 1:height(bestRuns)
    stateData = load(bestRuns.OptimizationRunFile(k),'runState');
    trackingData = load(bestRuns.TrackingDataFile(k),'tracking');
    fig = make_trajectory_figure(stateData.runState,trackingData.tracking);

    if saveFigures
        code = mission_code(bestRuns.Mission(k));
        stem = fullfile(figureDir,"pilot_best_trajectory_"+code);
        export_trajectory_figure(fig,stem);
        figureFiles(k,1) = stem+".eps";
        figureFiles(k,2) = stem+".png";
    end
end

fprintf('Centered Reviewer 2 trajectory plots complete.\n');
if saveFigures
    fprintf('Figures saved under:\n%s\n',figureDir);
end
end

function fig = make_trajectory_figure(runState,tracking)
truth = tracking.truth(:,1:3);
estimate = tracking.estimate(:,1:3);
mu = runState.settings.mu;
LU = runState.settings.LU;
moonCenter = [1-mu,0,0];
moonRadius = 1737.1/LU;
[xL1,xL2] = collinear_lagrange_points(mu);

% A slightly taller canvas and a deliberately centered inner plotting box
% are more reliable for MATLAB perspective axes than TightInset-based
% correction. The box is symmetric left/right and leaves a substantial
% lower margin for projected x/y labels, including long negative tick labels.
fig = figure('Color','w','Units','inches','Position',[1 1 7.6 7.0], ...
    'PaperUnits','inches','PaperSize',[7.6 7.0], ...
    'PaperPosition',[0 0 7.6 7.0],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');

plotPosition = [0.17 0.26 0.66 0.50];
ax = axes(fig,'Units','normalized','Position',plotPosition);
ax.PositionConstraint = 'innerposition';
hold(ax,'on');
box(ax,'on');
axis(ax,'equal');

hTruth = plot3(ax,truth(:,1),truth(:,2),truth(:,3), ...
    '--','Color',[0.55 0.55 0.55],'LineWidth',1.25, ...
    'DisplayName','Truth trajectory');
hEstimate = plot3(ax,estimate(:,1),estimate(:,2),estimate(:,3), ...
    '-','Color',[0.00 0.28 0.85],'LineWidth',2.9, ...
    'DisplayName','EKF estimate');

[sx,sy,sz] = sphere(30);
hMoon = surf(ax,moonCenter(1)+moonRadius*sx, ...
    moonCenter(2)+moonRadius*sy,moonCenter(3)+moonRadius*sz, ...
    'FaceColor',[0.72 0.72 0.72], ...
    'EdgeColor','none','FaceLighting','gouraud', ...
    'DisplayName','Moon');
camlight(ax,'headlight');
material(ax,'dull');

hL1 = plot3(ax,xL1,0,0,'^','MarkerSize',9, ...
    'MarkerFaceColor',[0.80 0.80 0.80], ...
    'MarkerEdgeColor','k','LineWidth',1.1,'DisplayName','L1');
hL2 = plot3(ax,xL2,0,0,'v','MarkerSize',9, ...
    'MarkerFaceColor',[0.80 0.80 0.80], ...
    'MarkerEdgeColor','k','LineWidth',1.1,'DisplayName','L2');
hStart = plot3(ax,truth(1,1),truth(1,2),truth(1,3),'o', ...
    'MarkerSize',8,'MarkerFaceColor',[0.20 0.70 0.25], ...
    'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','Start');
hEnd = plot3(ax,truth(end,1),truth(end,2),truth(end,3),'s', ...
    'MarkerSize',8,'MarkerFaceColor',[0.20 0.35 0.90], ...
    'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','End');

allPoints = [truth;estimate;moonCenter; ...
    moonCenter+[moonRadius 0 0];moonCenter-[moonRadius 0 0]; ...
    moonCenter+[0 moonRadius 0];moonCenter-[0 moonRadius 0]; ...
    moonCenter+[0 0 moonRadius];moonCenter-[0 0 moonRadius]; ...
    xL1 0 0;xL2 0 0];
xlim(ax,padded_limits(allPoints(:,1),0.08));
ylim(ax,padded_limits(allPoints(:,2),0.10));
zlim(ax,padded_limits(allPoints(:,3),0.10));
axis(ax,'vis3d');
ax.Projection = 'perspective';
view(ax,-37.5,30);
grid(ax,'off');

xlabel(ax,'x (LU)','FontWeight','bold');
ylabel(ax,'y (LU)','FontWeight','bold');
zlabel(ax,'z (LU)','FontWeight','bold');
set(ax,'FontName','Times New Roman','FontSize',12, ...
    'FontWeight','bold','LineWidth',1.2,'Layer','top');
ax.XLabel.FontSize = 14;
ax.YLabel.FontSize = 14;
ax.ZLabel.FontSize = 14;

% Create the legend, then explicitly restore the centered axes Position.
% This prevents northoutside legend layout from shifting/shrinking the 3-D
% axes differently for different trajectory geometries.
lgd = legend(ax,[hEstimate hTruth hMoon hL1 hL2 hStart hEnd], ...
    {'EKF estimate','Truth trajectory','Moon','L1','L2','Start','End'}, ...
    'Orientation','horizontal','NumColumns',4,'Box','on');
lgd.FontName = 'Times New Roman';
lgd.FontSize = 12;
lgd.FontWeight = 'bold';
lgd.ItemTokenSize = [18 10];
lgd.Units = 'normalized';
drawnow;
legendPosition = lgd.Position;
legendPosition(1) = 0.5-legendPosition(3)/2;
legendPosition(2) = 0.875;
lgd.Position = legendPosition;
lgd.AutoUpdate = 'off';

% Force the same centered plotting box after legend creation. This is the
% key step that keeps the projected axes centered on the page.
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
lowerValue = min(values);
upperValue = max(values);
span = upperValue-lowerValue;
if span <= 100*eps(max(1,max(abs(values))))
    span = max(0.02,0.05*max(1,abs(mean(values))));
end
padding = fraction*span;
limits = [lowerValue-padding,upperValue+padding];
end

function analysisDir = newest_analysis_directory(root)
assert(isfolder(root),'Comparison pilot root does not exist: %s',root);
directories = dir(fullfile(root,'FE_DATA_*'));
directories = directories([directories.isdir]);
assert(~isempty(directories), ...
    'No FE_DATA analysis directory was found under %s.',root);
[~,idx] = max([directories.datenum]);
analysisDir = fullfile(directories(idx).folder,directories(idx).name);
end

function export_trajectory_figure(fig,stem)
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

function code = mission_code(mission)
switch string(mission)
    case "LUNAR_GATEWAY"
        code = "lg";
    case "LOW_THRUST_TRANSFER"
        code = "lt";
    case "GATEWAY_IMPULSE"
        code = "gi";
    otherwise
        code = lower(string(mission));
end
end
