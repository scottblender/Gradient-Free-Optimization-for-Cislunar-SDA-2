from pathlib import Path
import re

path = Path('scripts/plot_study_definition_figures.m')
text = path.read_text()

visibility = r'''function outputs = create_visibility_keepout_figure(inspectFigure)
% Illustrate the unified minimum-angular-separation visibility geometry.
% The schematic is body-generic and intentionally contains only the
% geometry needed to explain physical occultation and sensor keepout.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

fig = publication_figure(7.2,5.1);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cBody = [0.70,0.70,0.70];
cOcc = [0.38,0.38,0.38];
cSensor = [0.92,0.55,0.05];
cLos = [0.05,0.05,0.05];
cOccShade = [0.82,0.82,0.82];
cSensorShade = [0.98,0.89,0.62];

observer = [-2.20,0.00];
body = [1.05,0.00];
bodyRadius = 0.66;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);

% Choose a representative threshold that exceeds the physical limb angle
% so the additional sensor-margin band is visible in the schematic.
thetaSensor = thetaOcc + deg2rad(8);
thetaKeepout = max(thetaOcc,thetaSensor);
thetaB = thetaKeepout + deg2rad(13);
targetRange = 4.75;
target = observer + targetRange*[cos(thetaB),sin(thetaB)];

ax = axes(fig,'Units','normalized','Position',[0.05,0.06,0.90,0.88]);
hold(ax,'on');
axis(ax,'equal');
axis(ax,'off');

bodyAngle = linspace(0,2*pi,240);
sectorRadius = 3.45;

% Physical occultation sector.
occAngles = linspace(-thetaOcc,thetaOcc,220);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(occAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(occAngles),observer(2)], ...
    cOccShade,'EdgeColor','none','HandleVisibility','off');

% Additional sensor-margin bands between the physical limb and the
% configured keepout boundary. Plot both sides of the exclusion cone.
upperMargin = linspace(thetaOcc,thetaKeepout,120);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(upperMargin),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(upperMargin),observer(2)], ...
    cSensorShade,'EdgeColor','none','HandleVisibility','off');
lowerMargin = linspace(-thetaKeepout,-thetaOcc,120);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(lowerMargin),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(lowerMargin),observer(2)], ...
    cSensorShade,'EdgeColor','none','HandleVisibility','off');

% Body and reference directions.
fill(ax,body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.2);
plot(ax,[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.3);

occEnd = observer + 3.65*[cos(thetaOcc),sin(thetaOcc)];
keepEnd = observer + 3.65*[cos(thetaKeepout),sin(thetaKeepout)];
plot(ax,[observer(1),occEnd(1)],[observer(2),occEnd(2)],':', ...
    'Color',cOcc,'LineWidth',1.9);
plot(ax,[observer(1),keepEnd(1)],[observer(2),keepEnd(2)],'--', ...
    'Color',cSensor,'LineWidth',2.1);
plot(ax,[observer(1),target(1)],[observer(2),target(2)],'-', ...
    'Color',cLos,'LineWidth',2.5);

plot(ax,observer(1),observer(2),'o','MarkerSize',11, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax,target(1),target(2),'o','MarkerSize',11, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

% Angle arcs.
draw_angle_arc_2d(ax,observer,0,thetaOcc,0.95,cOcc,1.9);
draw_angle_arc_2d(ax,observer,0,thetaKeepout,1.48,cSensor,2.1);
draw_angle_arc_2d(ax,observer,0,thetaB,2.10,cTarget,2.1);

% Object labels.
text(ax,observer(1)-0.02,observer(2)-0.46,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','center');
text(ax,target(1)+0.18,target(2)-0.02,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','left');
text(ax,body(1),body(2)-1.06,'Body b', ...
    'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','center');

% Angle labels and region callouts.
text(ax,-0.90,-0.22,'\theta_{occ,b}', ...
    'Color',cOcc,'FontWeight','bold','FontSize',16);
text(ax,-0.43,0.72,'\theta_{keepout,b}', ...
    'Color',cSensor,'FontWeight','bold','FontSize',16);
text(ax,0.03,1.40,'\theta_b', ...
    'Color',cTarget,'FontWeight','bold','FontSize',16);

ptOccRegion = observer + 2.05*[cos(-0.25*thetaOcc),sin(-0.25*thetaOcc)];
text(ax,-0.02,-1.48,{'physical';'occultation'}, ...
    'Color',cOcc,'FontSize',15,'FontAngle','italic', ...
    'FontWeight','bold','HorizontalAlignment','center');
quiver(ax,0.12,-1.14,ptOccRegion(1)-0.12,ptOccRegion(2)+1.14,0, ...
    'Color',cOcc,'LineWidth',1.3,'MaxHeadSize',0.18);

ptMargin = observer + 2.55*[cos(0.5*(thetaOcc+thetaKeepout)), ...
    sin(0.5*(thetaOcc+thetaKeepout))];
text(ax,0.78,2.05,{'sensor';'margin'}, ...
    'Color',cSensor,'FontSize',15,'FontAngle','italic', ...
    'FontWeight','bold','HorizontalAlignment','center');
quiver(ax,0.78,1.73,ptMargin(1)-0.78,ptMargin(2)-1.73,0, ...
    'Color',cSensor,'LineWidth',1.3,'MaxHeadSize',0.18);

xlim(ax,[-3.15,3.10]);
ylim(ax,[-2.00,3.45]);
set(findall(fig,'Type','text'),'FontName','Times New Roman');

figureFile = fullfile(outputDir,'visibility_keepout_geometry.eps');
inspect_before_export(fig,inspectFigure,'unified visibility / keepout geometry');
export_publication_eps(fig,figureFile);
close(fig);

outputs = struct();
outputs.figure = string(figureFile);
outputs.thetaSensor_deg = struct('earth',15,'moon',10,'sun',20);
fprintf('Saved visibility / keepout geometry to:\n  %s\n',figureFile);
end


'''

measurement = r'''function outputs = create_measurement_model_figure(inspectFigure)
% Illustrate the implemented angles-only RA/Dec relative LOS geometry.
% Visibility and keepout constraints intentionally do not appear here.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

fig = publication_figure(8.9,4.55);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cProjection = [0.42,0.42,0.42];
cAngle = [0.88,0.43,0.08];

rho = [3.15,2.10,1.75];
rhoXY = hypot(rho(1),rho(2));
alpha = atan2(rho(2),rho(1));
delta = asin(rho(3)/norm(rho));

% ---------------- (a) Right ascension ----------------
ax1 = axes(fig,'Units','normalized','Position',[0.06,0.12,0.39,0.80]);
hold(ax1,'on');
axis(ax1,'equal');
axis(ax1,'off');

projection = rho(1:2);
quiver(ax1,0,0,4.25,0,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
quiver(ax1,0,0,0,3.45,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
plot(ax1,[0,projection(1)],[0,projection(2)],'-k','LineWidth',2.2);
plot(ax1,0,0,'o','MarkerSize',10,'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax1,projection(1),projection(2),'o','MarkerSize',9,'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

alphaSamples = linspace(0,alpha,120);
alphaRadius = 1.20;
plot(ax1,alphaRadius*cos(alphaSamples),alphaRadius*sin(alphaSamples), ...
    '-','Color',cAngle,'LineWidth',2.0);

text(ax1,4.40,-0.08,'x','FontWeight','bold','FontSize',17);
text(ax1,-0.10,3.62,'y','FontWeight','bold','FontSize',17);
text(ax1,0,-0.43,'Observer','Color',cObserver, ...
    'FontWeight','bold','FontSize',15,'HorizontalAlignment','center');
text(ax1,projection(1)+0.10,projection(2)+0.24,'Target projection', ...
    'Color',cTarget,'FontWeight','bold','FontSize',15);
text(ax1,1.68,2.04,'\rho_{xy}', ...
    'Color',cProjection,'FontWeight','bold','FontSize',16);
text(ax1,1.52*cos(alpha/2),1.52*sin(alpha/2)+0.06,'\alpha', ...
    'Color',cAngle,'FontWeight','bold','FontSize',18);

title(ax1,'(a) Right ascension, \alpha', ...
    'FontSize',16,'FontWeight','bold');
xlim(ax1,[-0.72,4.65]);
ylim(ax1,[-0.72,3.85]);

% ---------------- (b) Declination ----------------
ax2 = axes(fig,'Units','normalized','Position',[0.55,0.12,0.39,0.80]);
hold(ax2,'on');
axis(ax2,'equal');
axis(ax2,'off');

target = [rhoXY,rho(3)];
quiver(ax2,0,0,4.35,0,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
quiver(ax2,0,0,0,3.25,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
plot(ax2,[0,target(1)],[0,target(2)],'-k','LineWidth',2.2);
plot(ax2,[target(1),target(1)],[0,target(2)],'--', ...
    'Color',cProjection,'LineWidth',1.5);
plot(ax2,0,0,'o','MarkerSize',10,'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax2,target(1),target(2),'o','MarkerSize',9,'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

deltaSamples = linspace(0,delta,120);
deltaRadius = 1.22;
plot(ax2,deltaRadius*cos(deltaSamples),deltaRadius*sin(deltaSamples), ...
    '-','Color',cAngle,'LineWidth',2.0);

% rho_xy is the horizontal coordinate in this panel, so it is an axis
% quantity and is labeled in black. Gray is reserved for projection aids.
text(ax2,2.28,0.28,'\rho_{xy}', ...
    'Color','k','FontWeight','bold','FontSize',16);
text(ax2,-0.12,3.42,'z','FontWeight','bold','FontSize',17);
text(ax2,0,-0.43,'Observer','Color',cObserver, ...
    'FontWeight','bold','FontSize',15,'HorizontalAlignment','center');
text(ax2,target(1)+0.10,target(2)+0.22,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',15);
text(ax2,2.02,1.88,'\rho','FontWeight','bold','FontSize',16);
text(ax2,1.56*cos(delta/2),1.56*sin(delta/2)+0.06,'\delta', ...
    'Color',cAngle,'FontWeight','bold','FontSize',18);

title(ax2,'(b) Declination, \delta', ...
    'FontSize',16,'FontWeight','bold');
xlim(ax2,[-0.72,4.75]);
ylim(ax2,[-0.72,3.60]);

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


'''

pattern_visibility = re.compile(r'function outputs = create_visibility_keepout_figure\(inspectFigure\).*?(?=function outputs = create_measurement_model_figure\(inspectFigure\))', re.S)
pattern_measurement = re.compile(r'function outputs = create_measurement_model_figure\(inspectFigure\).*?(?=function \[fig,axesHandles,textAx\] = schematic_figure_layout|function draw_angle_arc_2d)', re.S)

if not pattern_visibility.search(text):
    raise RuntimeError('Could not locate visibility figure function.')
if not pattern_measurement.search(text):
    raise RuntimeError('Could not locate measurement figure function.')

text = pattern_visibility.sub(lambda m: visibility, text, count=1)
text = pattern_measurement.sub(lambda m: measurement, text, count=1)

assert 'thetaSensor = thetaOcc + deg2rad(8);' in text
assert "'Color','k','FontWeight','bold','FontSize',16" in text
assert 'Configured center-separation limits' not in text
assert 'Admissible LOS:' not in text

path.write_text(text)
print('Applied final schematic cleanup v2.')
