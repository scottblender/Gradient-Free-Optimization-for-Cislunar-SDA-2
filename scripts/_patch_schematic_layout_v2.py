from pathlib import Path
import re

path = Path('scripts/plot_study_definition_figures.m')
text = path.read_text()

visibility = r'''function outputs = create_visibility_keepout_figure(inspectFigure)
% Illustrate the unified Earth/Moon/Sun visibility framework.
% Physical occultation and configured exclusion use the same observer-
% centered angular separation geometry. For each body b,
% theta_keepout,b = max(theta_occ,b,theta_sensor,b).

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

[fig,ax,textAx] = schematic_figure_layout(2,4.35,4);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cBody = [0.70,0.70,0.70];
cOcc = [0.38,0.38,0.38];
cSensor = [0.92,0.55,0.05];
cKeepout = [0.72,0.13,0.12];
cOccShade = [0.94,0.94,0.94];
cSensorShade = [1.00,0.95,0.87];

% ---- (a) Unified minimum angular separation ----
hold(ax(1),'on');
axis(ax(1),'equal');
axis(ax(1),'off');

observer = [-2.45,0.0];
body = [0.60,0.0];
bodyRadius = 0.55;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);
thetaSensor = deg2rad(24);
thetaKeepout = max(thetaOcc,thetaSensor);
thetaTarget = deg2rad(42);
targetRange = 3.95;
target = observer + targetRange*[cos(thetaTarget),sin(thetaTarget)];

bodyAngle = linspace(0,2*pi,240);

% Draw the complete keepout sector first, then overlay the physical
% occultation sector so the configured angular margin is visually distinct.
sectorRadius = 2.30;
keepoutAngles = linspace(-thetaKeepout,thetaKeepout,240);
patch(ax(1), ...
    [observer(1),observer(1)+sectorRadius*cos(keepoutAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(keepoutAngles),observer(2)], ...
    cSensorShade,'EdgeColor','none','HandleVisibility','off');
occAngles = linspace(-thetaOcc,thetaOcc,180);
patch(ax(1), ...
    [observer(1),observer(1)+sectorRadius*cos(occAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(occAngles),observer(2)], ...
    cOccShade,'EdgeColor','none','HandleVisibility','off');

fill(ax(1),body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.1);
plot(ax(1),[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.2);

occEnd = observer + 3.20*[cos(thetaOcc),sin(thetaOcc)];
keepoutEnd = observer + 3.20*[cos(thetaKeepout),sin(thetaKeepout)];
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

draw_angle_arc_2d(ax(1),observer,0,thetaOcc,0.72,cOcc,1.5);
draw_angle_arc_2d(ax(1),observer,0,thetaKeepout,1.12,cSensor,1.6);
draw_angle_arc_2d(ax(1),observer,0,thetaTarget,1.58,cKeepout,1.7);

text(ax(1),observer(1),observer(2)-0.36,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(1),target(1)+0.02,target(2)+0.18,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(1),body(1),body(2)-0.10,'Body b', ...
    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');

text(ax(1),observer(1)+0.82*cos(thetaOcc/2), ...
    observer(2)+0.82*sin(thetaOcc/2)-0.15,'\theta_{occ,b}', ...
    'Color',cOcc,'FontWeight','bold','FontSize',11);
text(ax(1),observer(1)+1.25*cos(thetaKeepout/2), ...
    observer(2)+1.25*sin(thetaKeepout/2)+0.05, ...
    '\theta_{keepout,b}','Color',cSensor, ...
    'FontWeight','bold','FontSize',11);
text(ax(1),observer(1)+1.73*cos(thetaTarget/2), ...
    observer(2)+1.73*sin(thetaTarget/2)+0.06,'\theta_b', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',11);

title(ax(1),'(a) Unified minimum angular separation', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(1),[-2.90,3.00]);
ylim(ax(1),[-1.25,3.05]);

% ---- (b) Zero configured-threshold limit ----
hold(ax(2),'on');
axis(ax(2),'equal');
axis(ax(2),'off');

observer = [-2.45,0.0];
body = [0.60,0.0];
bodyRadius = 0.55;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);
thetaSensor = 0;
thetaKeepout = max(thetaOcc,thetaSensor);
thetaTarget = deg2rad(36);
targetRange = 4.00;
target = observer + targetRange*[cos(thetaTarget),sin(thetaTarget)];

sectorRadius = 2.30;
occAngles = linspace(-thetaOcc,thetaOcc,180);
patch(ax(2), ...
    [observer(1),observer(1)+sectorRadius*cos(occAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(occAngles),observer(2)], ...
    cOccShade,'EdgeColor','none','HandleVisibility','off');

fill(ax(2),body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.1);
plot(ax(2),[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.2);

boundaryEnd = observer + 3.20*[cos(thetaOcc),sin(thetaOcc)];
plot(ax(2),[observer(1),boundaryEnd(1)], ...
    [observer(2),boundaryEnd(2)],'--', ...
    'Color',cKeepout,'LineWidth',1.8);
plot(ax(2),[observer(1),target(1)],[observer(2),target(2)],'-k', ...
    'LineWidth',2.2);

plot(ax(2),observer(1),observer(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax(2),target(1),target(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

draw_angle_arc_2d(ax(2),observer,0,thetaKeepout,0.94,cKeepout,1.7);
draw_angle_arc_2d(ax(2),observer,0,thetaTarget,1.58,cKeepout,1.7);

text(ax(2),observer(1),observer(2)-0.36,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(2),target(1)+0.02,target(2)+0.18,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(2),body(1),body(2)-0.10,'Body b', ...
    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');
text(ax(2),observer(1)+1.08*cos(thetaOcc/2), ...
    observer(2)+1.08*sin(thetaOcc/2)+0.08, ...
    '\theta_{keepout,b}=\theta_{occ,b}', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',10.8);
text(ax(2),observer(1)+1.73*cos(thetaTarget/2), ...
    observer(2)+1.73*sin(thetaTarget/2)+0.06,'\theta_b', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',11);

title(ax(2),'(b) Zero sensor-threshold limit', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(2),[-2.90,3.00]);
ylim(ax(2),[-1.25,3.05]);

% All visibility explanation belongs on this figure. The footer has a
% physical height determined by the number of rows, so the equations remain
% separated from the geometry even when the figure dimensions change.
axis(textAx,'off');
y = linspace(0.86,0.14,4);
text(textAx,0.02,y(1), ...
    '\theta_{keepout,b}=max(\theta_{occ,b},\theta_{sensor,b}),   b \in {Earth, Moon, Sun}', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',12,'Color',cKeepout, ...
    'VerticalAlignment','middle');
text(textAx,0.02,y(2), ...
    'Implemented visibility: \theta_b>\theta_{occ,b} and \theta_b\geq\theta_{sensor,b}; tangency is blocked.', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',10.8,'VerticalAlignment','middle');
text(textAx,0.02,y(3), ...
    'Configured center-separation limits: Earth 15^{\circ}, Moon 10^{\circ}, Sun 20^{\circ}.', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','normal','FontSize',10.8,'VerticalAlignment','middle');
text(textAx,0.02,y(4), ...
    'As \theta_{sensor,b}\rightarrow0, \theta_{keepout,b}\rightarrow\theta_{occ,b}; exclusion reduces to physical occlusion.', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','normal','FontSize',10.8,'VerticalAlignment','middle');

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
% Illustrate the implemented angles-only relative LOS measurement.
% This figure intentionally contains only measurement geometry; visibility
% and keepout logic are documented in visibility_keepout_geometry.eps.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

[fig,ax,textAx] = schematic_figure_layout(2,4.05,2);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cProjection = [0.42,0.42,0.42];
cAngle = [0.88,0.43,0.08];

% Use one internally consistent LOS vector in both 2-D projections.
rho = [3.15,2.10,1.75];
rhoXY = hypot(rho(1),rho(2));
alpha = atan2(rho(2),rho(1));
delta = asin(rho(3)/norm(rho));

% ---- (a) Right ascension in the x-y plane ----
hold(ax(1),'on');
axis(ax(1),'equal');
axis(ax(1),'off');

projection = rho(1:2);
quiver(ax(1),0,0,4.25,0,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
quiver(ax(1),0,0,0,3.45,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
plot(ax(1),[0,projection(1)],[0,projection(2)],'-k','LineWidth',2.2);
plot(ax(1),0,0,'o','MarkerSize',10, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax(1),projection(1),projection(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

alphaSamples = linspace(0,alpha,120);
alphaRadius = 1.15;
plot(ax(1),alphaRadius*cos(alphaSamples), ...
    alphaRadius*sin(alphaSamples),'-','Color',cAngle,'LineWidth',2.0);

text(ax(1),4.38,-0.08,'x','FontWeight','bold','FontSize',14);
text(ax(1),-0.10,3.62,'y','FontWeight','bold','FontSize',14);
text(ax(1),0,-0.38,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(1),projection(1)+0.05,projection(2)+0.22,'LOS projection', ...
    'Color',cTarget,'FontWeight','bold','FontSize',11, ...
    'HorizontalAlignment','left');
text(ax(1),1.72,1.26,'\rho_{xy}', ...
    'Color',cProjection,'FontWeight','bold','FontSize',12);
text(ax(1),1.32*cos(alpha/2),1.32*sin(alpha/2)+0.03,'\alpha', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);

title(ax(1),'(a) Right ascension in the x-y plane', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(1),[-0.72,4.65]);
ylim(ax(1),[-0.70,3.85]);

% ---- (b) Declination in the vertical LOS plane ----
hold(ax(2),'on');
axis(ax(2),'equal');
axis(ax(2),'off');

quiver(ax(2),0,0,4.35,0,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
quiver(ax(2),0,0,0,3.15,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.08);
plot(ax(2),[0,rhoXY],[0,rho(3)],'-k','LineWidth',2.2);
plot(ax(2),[rhoXY,rhoXY],[0,rho(3)],'--', ...
    'Color',cProjection,'LineWidth',1.3);
plot(ax(2),[0,rhoXY],[0,0],'--', ...
    'Color',cProjection,'LineWidth',1.3);
plot(ax(2),0,0,'o','MarkerSize',10, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax(2),rhoXY,rho(3),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

deltaSamples = linspace(0,delta,120);
deltaRadius = 1.18;
plot(ax(2),deltaRadius*cos(deltaSamples), ...
    deltaRadius*sin(deltaSamples),'-','Color',cAngle,'LineWidth',2.0);

text(ax(2),4.48,-0.08,'\rho_{xy}','Color',cProjection, ...
    'FontWeight','bold','FontSize',13);
text(ax(2),-0.10,3.32,'z','FontWeight','bold','FontSize',14);
text(ax(2),0,-0.38,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax(2),rhoXY+0.08,rho(3)+0.20,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12);
text(ax(2),2.18,1.12,'\rho','FontWeight','bold','FontSize',12);
text(ax(2),1.34*cos(delta/2),1.34*sin(delta/2)+0.03,'\delta', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);

title(ax(2),'(b) Declination above the x-y plane', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax(2),[-0.72,4.72]);
ylim(ax(2),[-0.70,3.55]);

% Measurement equations only. No visibility text is placed on this figure.
axis(textAx,'off');
text(textAx,0.50,0.73,'\rho = r_{tar}-r_{obs}', ...
    'Units','normalized','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',12);
text(textAx,0.25,0.27,'\alpha = atan2(\rho_y,\rho_x)', ...
    'Units','normalized','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',11.5);
text(textAx,0.75,0.27,'\delta = asin(\rho_z/||\rho||)', ...
    'Units','normalized','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',11.5);

set(findall(fig,'Type','text'),'FontName','Times New Roman');

figureFile = fullfile(outputDir,'measurement_model_radec_geometry.eps');
inspect_before_export(fig,inspectFigure,'RA/Dec measurement geometry');
export_publication_eps(fig,figureFile);
close(fig);

outputs = struct();
outputs.figure = string(figureFile);
outputs.measurementType = "ANGLES_ONLY";
fprintf('Saved RA/Dec measurement geometry to:\n  %s\n',figureFile);
end'''

helper = r'''function [fig,axesHandles,textAx] = schematic_figure_layout( ...
    numPanels,geometryHeightInches,textLineCount)
%SCHEMATIC_FIGURE_LAYOUT Create non-overlapping geometry and text regions.
% Width grows with panel count. Height grows with geometry plus a footer
% whose physical height is proportional to the requested text-row count.

validateattributes(numPanels,{'numeric'},{'scalar','integer','positive'});
validateattributes(geometryHeightInches,{'numeric'}, ...
    {'scalar','real','finite','positive'});
validateattributes(textLineCount,{'numeric'}, ...
    {'scalar','integer','nonnegative'});

panelWidthInches = 4.35;
sideMarginInches = 0.42;
panelGapInches = 0.42;
topMarginInches = 0.42;
bottomMarginInches = 0.18;
footerLineInches = 0.42;
footerPaddingInches = 0.28;

if textLineCount > 0
    footerHeightInches = footerPaddingInches + ...
        textLineCount*footerLineInches;
else
    footerHeightInches = 0;
end

widthInches = 2*sideMarginInches + ...
    numPanels*panelWidthInches + (numPanels-1)*panelGapInches;
heightInches = bottomMarginInches + footerHeightInches + ...
    geometryHeightInches + topMarginInches;

fig = publication_figure(widthInches,heightInches);

geometryBottomInches = bottomMarginInches + footerHeightInches;
axesHandles = gobjects(numPanels,1);
for k = 1:numPanels
    leftInches = sideMarginInches + ...
        (k-1)*(panelWidthInches+panelGapInches);
    axesHandles(k) = axes(fig,'Units','normalized','Position',[ ...
        leftInches/widthInches, ...
        geometryBottomInches/heightInches, ...
        panelWidthInches/widthInches, ...
        geometryHeightInches/heightInches]);
    axesHandles(k).PositionConstraint = 'innerposition';
end

if textLineCount > 0
    textAx = axes(fig,'Units','normalized','Position',[ ...
        sideMarginInches/widthInches, ...
        bottomMarginInches/heightInches, ...
        (widthInches-2*sideMarginInches)/widthInches, ...
        footerHeightInches/heightInches]);
    textAx.PositionConstraint = 'innerposition';
    xlim(textAx,[0,1]);
    ylim(textAx,[0,1]);
    axis(textAx,'off');
else
    textAx = gobjects(1);
end
end'''

pattern_visibility = re.compile(
    r'function outputs = create_visibility_keepout_figure\(inspectFigure\).*?(?=\nfunction outputs = create_measurement_model_figure\(inspectFigure\))',
    re.S)
pattern_measurement = re.compile(
    r'function outputs = create_measurement_model_figure\(inspectFigure\).*?(?=\nfunction \[fig,axesHandles,textAx\] = schematic_figure_layout\()',
    re.S)
pattern_helper = re.compile(
    r'function \[fig,axesHandles,textAx\] = schematic_figure_layout\(.*?(?=\nfunction draw_angle_arc_2d\()',
    re.S)

text, n1 = pattern_visibility.subn(lambda m: visibility + '\n\n', text, count=1)
text, n2 = pattern_measurement.subn(lambda m: measurement + '\n\n', text, count=1)
text, n3 = pattern_helper.subn(lambda m: helper + '\n\n', text, count=1)

if n1 != 1:
    raise SystemExit(f'Expected one visibility function replacement, got {n1}.')
if n2 != 1:
    raise SystemExit(f'Expected one measurement function replacement, got {n2}.')
if n3 != 1:
    raise SystemExit(f'Expected one schematic helper replacement, got {n3}.')

measurement_block = text.split('function outputs = create_measurement_model_figure',1)[1]
measurement_block = measurement_block.split('function [fig,axesHandles,textAx] = schematic_figure_layout',1)[0]
for forbidden in ['theta_{keepout', 'Earth, Moon, Sun', 'Visibility gate', 'visibility gate']:
    if forbidden in measurement_block:
        raise SystemExit(f'Visibility text leaked into measurement figure: {forbidden}')

visibility_block = text.split('function outputs = create_visibility_keepout_figure',1)[1]
visibility_block = visibility_block.split('function outputs = create_measurement_model_figure',1)[0]
required = [
    'theta_{keepout,b}=max',
    'Earth 15^{\\circ}',
    'Moon 10^{\\circ}',
    'Sun 20^{\\circ}',
    'theta_{sensor,b}\\rightarrow0',
    'tangency is blocked',
]
for item in required:
    if item not in visibility_block:
        raise SystemExit(f'Missing expected visibility content: {item}')

path.write_text(text)

Path('scripts/_patch_schematic_layout_v2.py').unlink(missing_ok=True)
Path('.github/workflows/one-time-schematic-layout-v2.yml').unlink(missing_ok=True)
