from pathlib import Path
import re

path = Path('scripts/plot_study_definition_figures.m')
text = path.read_text()

visibility = r'''function outputs = create_visibility_keepout_figure(inspectFigure)
% Illustrate the unified Earth/Moon/Sun visibility framework.
% Physical occultation and configured exclusion are represented by the
% same observer-centered angular separation geometry. For body b,
% theta_keepout,b = max(theta_occ,b,theta_sensor,b).

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

% The canvas height is computed from the amount of explanatory text so the
% equations never have to compete with the geometry for plotting space.
[fig,ax,textAx] = schematic_figure_layout(2,4.2,4);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cBody = [0.68,0.68,0.68];
cOcc = [0.35,0.35,0.35];
cSensor = [0.92,0.55,0.05];
cKeepout = [0.72,0.13,0.12];
cShade = [1.00,0.95,0.87];

% ---- (a) Unified minimum angular separation ----
hold(ax(1),'on');
axis(ax(1),'equal');
axis(ax(1),'off');

observer = [-2.45,0.00];
body = [0.20,0.00];
bodyRadius = 0.55;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);
thetaSensor = deg2rad(24);
thetaKeepout = max(thetaOcc,thetaSensor);
thetaTarget = deg2rad(40);
target = observer + 4.55*[cos(thetaTarget),sin(thetaTarget)];

% Shade the complete forbidden cone associated with theta_keepout.
sectorAngle = linspace(-thetaKeepout,thetaKeepout,240);
sectorRadius = 2.25;
patch(ax(1), ...
    [observer(1),observer(1)+sectorRadius*cos(sectorAngle),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(sectorAngle),observer(2)], ...
    cShade,'EdgeColor','none','HandleVisibility','off');

bodyAngle = linspace(0,2*pi,240);
fill(ax(1),body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.0);

% Body-center direction, physical tangent, configured keepout boundary,
% and an accepted target line of sight.
plot(ax(1),[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.2);
occEnd = observer + 3.15*[cos(thetaOcc),sin(thetaOcc)];
keepoutEnd = observer + 3.15*[cos(thetaKeepout),sin(thetaKeepout)];
plot(ax(1),[observer(1),occEnd(1)],[observer(2),occEnd(2)],':', ...
    'Color',cOcc,'LineWidth',1.6);
plot(ax(1),[observer(1),keepoutEnd(1)], ...
    [observer(2),keepoutEnd(2)],'--', ...
    'Color',cSensor,'LineWidth',1.8);
plot(ax(1),[observer(1),target(1)],[observer(2),target(2)],'-k', ...
    'LineWidth',2.2);

plot(ax(1),observer(1),observer(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax(1),target(1),target(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

draw_angle_arc_2d(ax(1),observer,0,thetaOcc,0.66,cOcc,1.5);
draw_angle_arc_2d(ax(1),observer,thetaOcc,thetaKeepout,0.98,cSensor,1.6);
draw_angle_arc_2d(ax(1),observer,0,thetaTarget,1.42,cKeepout,1.7);

text(ax(1),observer(1),observer(2)-0.35,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(1),target(1)+0.02,target(2)+0.22,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(1),body(1),body(2)-0.10,'Body b', ...
    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');

text(ax(1),observer(1)+0.58*cos(thetaOcc/2), ...
    observer(2)+0.58*sin(thetaOcc/2)-0.14,'\theta_{occ,b}', ...
    'Color',cOcc,'FontWeight','bold','FontSize',11);
text(ax(1),observer(1)+1.06*cos((thetaOcc+thetaKeepout)/2), ...
    observer(2)+1.06*sin((thetaOcc+thetaKeepout)/2)+0.05, ...
    '\theta_{keepout,b}','Color',cSensor, ...
    'FontWeight','bold','FontSize',11);
text(ax(1),observer(1)+1.50*cos(thetaTarget/2), ...
    observer(2)+1.50*sin(thetaTarget/2)+0.05,'\theta_b', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',11);

text(ax(1),0.22,-0.79,'b \in {Earth, Moon, Sun}', ...
    'FontWeight','bold','FontSize',10.5,'HorizontalAlignment','center');

title(ax(1),'(a) Unified minimum angular separation', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(1),[-2.85,3.10]);
ylim(ax(1),[-1.30,2.35]);

% ---- (b) Zero sensor-threshold limit ----
hold(ax(2),'on');
axis(ax(2),'equal');
axis(ax(2),'off');

observer = [-2.45,0.00];
body = [0.20,0.00];
bodyRadius = 0.55;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);
thetaSensor = 0;
thetaKeepout = max(thetaOcc,thetaSensor);
thetaTarget = deg2rad(35);
target = observer + 4.50*[cos(thetaTarget),sin(thetaTarget)];

sectorAngle = linspace(-thetaKeepout,thetaKeepout,240);
sectorRadius = 2.25;
patch(ax(2), ...
    [observer(1),observer(1)+sectorRadius*cos(sectorAngle),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(sectorAngle),observer(2)], ...
    [0.94,0.94,0.94],'EdgeColor','none','HandleVisibility','off');

fill(ax(2),body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.0);
plot(ax(2),[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.2);
boundaryEnd = observer + 3.15*[cos(thetaOcc),sin(thetaOcc)];
plot(ax(2),[observer(1),boundaryEnd(1)], ...
    [observer(2),boundaryEnd(2)],'--', ...
    'Color',cKeepout,'LineWidth',1.8);
plot(ax(2),[observer(1),target(1)],[observer(2),target(2)],'-k', ...
    'LineWidth',2.2);

plot(ax(2),observer(1),observer(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax(2),target(1),target(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

draw_angle_arc_2d(ax(2),observer,0,thetaOcc,0.88,cKeepout,1.7);
draw_angle_arc_2d(ax(2),observer,0,thetaTarget,1.42,cKeepout,1.7);

text(ax(2),observer(1),observer(2)-0.35,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(2),target(1)+0.02,target(2)+0.22,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(2),body(1),body(2)-0.10,'Body b', ...
    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');
text(ax(2),observer(1)+0.92*cos(thetaOcc/2), ...
    observer(2)+0.92*sin(thetaOcc/2)+0.08, ...
    '\theta_{keepout,b}=\theta_{occ,b}', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',10.8);
text(ax(2),observer(1)+1.50*cos(thetaTarget/2), ...
    observer(2)+1.50*sin(thetaTarget/2)+0.05,'\theta_b', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',11);

title(ax(2),'(b) Zero-threshold limit', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(2),[-2.85,3.10]);
ylim(ax(2),[-1.30,2.35]);

% Dedicated explanation band. Keeping these equations outside either data
% axes prevents labels from covering the geometry as the figure is resized.
axis(textAx,'off');
text(textAx,0.00,0.88, ...
    '\theta_{keepout,b}=max(\theta_{occ,b},\theta_{sensor,b}),   b \in {Earth, Moon, Sun}', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',12,'Color',cKeepout, ...
    'VerticalAlignment','top');
text(textAx,0.00,0.59, ...
    'Visible when \theta_b>\theta_{occ,b} and \theta_b\geq\theta_{keepout,b}.', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',11,'VerticalAlignment','top');
text(textAx,0.00,0.31, ...
    'Earth, Moon, and Sun use the same test; configured sensor thresholds are 15^{\circ}, 10^{\circ}, and 20^{\circ}, respectively.', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','normal','FontSize',10.5,'VerticalAlignment','top');
text(textAx,0.00,0.06, ...
    'As \theta_{sensor,b}\rightarrow0, \theta_{keepout,b}\rightarrow\theta_{occ,b}; the exclusion constraint reduces to physical occlusion.', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','normal','FontSize',10.5,'VerticalAlignment','bottom');

set(findall(fig,'Type','text'),'FontName','Times New Roman');

figureFile = fullfile(outputDir,'visibility_keepout_geometry.eps');
inspect_before_export(fig,inspectFigure,'unified visibility / keepout geometry');
export_publication_eps(fig,figureFile);
close(fig);

outputs = struct();
outputs.figure = string(figureFile);
outputs.thetaSensor_deg = struct('earth',15,'moon',10,'sun',20);
fprintf('Saved visibility / keepout geometry to:\n  %s\n',figureFile);
end'''

measurement = r'''function outputs = create_measurement_model_figure(inspectFigure)
% Illustrate the implemented angles-only relative LOS measurement using two
% 2-D projections. Separating right ascension and declination removes the
% perspective ambiguity of the previous 3-D schematic.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

[fig,ax,textAx] = schematic_figure_layout(2,4.1,4);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cProjection = [0.42,0.42,0.42];
cAngle = [0.88,0.43,0.08];
cKeepout = [0.72,0.13,0.12];

% Use one internally consistent LOS vector for both projections.
rho = [3.15,2.10,1.75];
rhoXY = hypot(rho(1),rho(2));
alpha = atan2(rho(2),rho(1));
delta = asin(rho(3)/norm(rho));

% ---- (a) Right ascension: x-y projection ----
hold(ax(1),'on');
axis(ax(1),'equal');
axis(ax(1),'off');

origin = [0,0];
projection = rho(1:2);
axisLength = 4.25;
quiver(ax(1),0,0,axisLength,0,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
quiver(ax(1),0,0,0,3.45,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
plot(ax(1),[0,projection(1)],[0,projection(2)],'-k','LineWidth',2.2);
plot(ax(1),origin(1),origin(2),'o','MarkerSize',10, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax(1),projection(1),projection(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

alphaSamples = linspace(0,alpha,120);
alphaRadius = 1.15;
plot(ax(1),alphaRadius*cos(alphaSamples), ...
    alphaRadius*sin(alphaSamples),'-','Color',cAngle,'LineWidth',2.0);

text(ax(1),4.38,-0.08,'x','FontWeight','bold','FontSize',14);
text(ax(1),-0.10,3.63,'y','FontWeight','bold','FontSize',14);
text(ax(1),-0.10,-0.35,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(1),projection(1)+0.08,projection(2)+0.20, ...
    'Target projection','Color',cTarget,'FontWeight','bold','FontSize',11);
text(ax(1),1.72,1.23,'\rho_{xy}', ...
    'Color',cProjection,'FontWeight','bold','FontSize',12);
text(ax(1),1.30*cos(alpha/2),1.30*sin(alpha/2)+0.03,'\alpha', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);

title(ax(1),'(a) Right ascension in the x-y plane', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(1),[-0.75,4.65]);
ylim(ax(1),[-0.75,3.90]);

% ---- (b) Declination: vertical plane containing the LOS ----
hold(ax(2),'on');
axis(ax(2),'equal');
axis(ax(2),'off');

origin = [0,0];
target = [rhoXY,rho(3)];
axisLength = 4.35;
quiver(ax(2),0,0,axisLength,0,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
quiver(ax(2),0,0,0,3.25,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
plot(ax(2),[0,target(1)],[0,target(2)],'-k','LineWidth',2.2);
plot(ax(2),[target(1),target(1)],[0,target(2)],'--', ...
    'Color',cProjection,'LineWidth',1.4);
plot(ax(2),[0,target(1)],[0,0],'--', ...
    'Color',cProjection,'LineWidth',1.4);
plot(ax(2),origin(1),origin(2),'o','MarkerSize',10, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax(2),target(1),target(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

% Declination is the elevation of rho above its x-y projection.
deltaSamples = linspace(0,delta,120);
deltaRadius = 1.22;
plot(ax(2),deltaRadius*cos(deltaSamples), ...
    deltaRadius*sin(deltaSamples),'-','Color',cAngle,'LineWidth',2.0);

text(ax(2),4.48,-0.08,'\rho_{xy}','Color',cProjection, ...
    'FontWeight','bold','FontSize',12);
text(ax(2),-0.12,3.42,'z','FontWeight','bold','FontSize',14);
text(ax(2),-0.10,-0.35,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(2),target(1)+0.10,target(2)+0.18,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12);
text(ax(2),2.10,1.10,'\rho','FontWeight','bold','FontSize',13);
text(ax(2),1.38*cos(delta/2),1.38*sin(delta/2)+0.03,'\delta', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);

title(ax(2),'(b) Declination above the x-y plane', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(2),[-0.75,4.75]);
ylim(ax(2),[-0.75,3.65]);

% Dedicated equation band sized independently from the two geometry axes.
axis(textAx,'off');
text(textAx,0.00,0.88, ...
    '\rho=r_{tar}-r_{obs}', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',12,'VerticalAlignment','top');
text(textAx,0.25,0.88, ...
    '\alpha=atan2(\rho_y,\rho_x)', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',12,'VerticalAlignment','top');
text(textAx,0.57,0.88, ...
    '\delta=asin(\rho_z/||\rho||)', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',12,'VerticalAlignment','top');
text(textAx,0.00,0.47, ...
    'Unified visibility gate is evaluated before a measurement is formed.', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','normal','FontSize',10.8,'VerticalAlignment','top');
text(textAx,0.00,0.15, ...
    '\theta_b>\theta_{occ,b},   \theta_b\geq\theta_{keepout,b},   \theta_{keepout,b}=max(\theta_{occ,b},\theta_{sensor,b}),   b \in {Earth, Moon, Sun}', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',10.5,'Color',cKeepout, ...
    'VerticalAlignment','bottom');

set(findall(fig,'Type','text'),'FontName','Times New Roman');

figureFile = fullfile(outputDir,'measurement_model_radec_geometry.eps');
inspect_before_export(fig,inspectFigure,'RA/Dec measurement geometry');
export_publication_eps(fig,figureFile);
close(fig);

outputs = struct();
outputs.figure = string(figureFile);
outputs.measurementType = "ANGLES_ONLY";
fprintf('Saved RA/Dec measurement geometry to:\n  %s\n',figureFile);
end


function [fig,axesHandles,textAx] = schematic_figure_layout( ...
    numPanels,geometryHeightInches,textLineCount)
%SCHEMATIC_FIGURE_LAYOUT Build a resize-safe schematic canvas.
% Figure width scales with panel count and figure height scales with the
% number of explanatory text rows. Geometry and prose occupy separate
% normalized regions, preventing text from overlapping plotted objects.

validateattributes(numPanels,{'numeric'},{'scalar','integer','positive'});
validateattributes(geometryHeightInches,{'numeric'},{'scalar','positive'});
validateattributes(textLineCount,{'numeric'},{'scalar','integer','nonnegative'});

panelWidthInches = 4.25;
sidePaddingInches = 0.65;
textHeightInches = max(0.95,0.26*textLineCount+0.35);
widthInches = max(7.2,numPanels*panelWidthInches+sidePaddingInches);
heightInches = geometryHeightInches+textHeightInches;

fig = publication_figure(widthInches,heightInches);

leftMargin = 0.055;
rightMargin = 0.035;
panelGap = 0.055;
bottomMargin = 0.035;
topMargin = 0.075;
textFraction = textHeightInches/heightInches;
textTop = bottomMargin+textFraction;
geometryBottom = textTop+0.035;
geometryHeight = 1-topMargin-geometryBottom;
panelWidth = (1-leftMargin-rightMargin-(numPanels-1)*panelGap)/numPanels;

axesHandles = gobjects(numPanels,1);
for k = 1:numPanels
    left = leftMargin+(k-1)*(panelWidth+panelGap);
    axesHandles(k) = axes(fig,'Units','normalized', ...
        'Position',[left,geometryBottom,panelWidth,geometryHeight]);
end

textAx = axes(fig,'Units','normalized', ...
    'Position',[leftMargin,bottomMargin, ...
    1-leftMargin-rightMargin,textFraction-0.02]);
axis(textAx,'off');
end'''

pattern_visibility = re.compile(
    r'function outputs = create_visibility_keepout_figure\(inspectFigure\).*?(?=\nfunction outputs = create_measurement_model_figure\(inspectFigure\))',
    re.S)
pattern_measurement = re.compile(
    r'function outputs = create_measurement_model_figure\(inspectFigure\).*?(?=\nfunction draw_angle_arc_2d\()',
    re.S)

text, n1 = pattern_visibility.subn(lambda match: visibility + '\n\n', text, count=1)
if n1 != 1:
    raise SystemExit(f'Expected one visibility function replacement, got {n1}.')

text, n2 = pattern_measurement.subn(lambda match: measurement + '\n\n', text, count=1)
if n2 != 1:
    raise SystemExit(f'Expected one measurement function replacement, got {n2}.')

required = [
    'function [fig,axesHandles,textAx] = schematic_figure_layout',
    'Right ascension in the x-y plane',
    'Declination above the x-y plane',
    'thetaKeepout = max(thetaOcc,thetaSensor);',
    "b \\in {Earth, Moon, Sun}",
    'numPairOrbitsPerFamily = 16;',
    'numDroOrbits = 16;',
]
for item in required:
    if item not in text:
        raise SystemExit(f'Missing expected text: {item}')

path.write_text(text)

# Remove one-time patch machinery from the production commit.
Path('scripts/_patch_dynamic_schematics.py').unlink(missing_ok=True)
Path('.github/workflows/one-time-dynamic-schematics.yml').unlink(missing_ok=True)
