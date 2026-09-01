from pathlib import Path

path = Path('scripts/plot_study_definition_figures.m')
text = path.read_text()

start_vis = text.index('function outputs = create_visibility_keepout_figure(inspectFigure)')
start_meas = text.index('function outputs = create_measurement_model_figure(inspectFigure)')
start_helper = text.index('function [fig,axesHandles,textAx] = schematic_figure_layout')

visibility = r'''function outputs = create_visibility_keepout_figure(inspectFigure)
% Illustrate the unified Earth/Moon/Sun angular-separation framework.
% The geometry is intentionally body-generic: physical occultation and
% configured sensor exclusion are combined through one keepout threshold.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

% Size the canvas from the amount of content. Geometry, the body-threshold
% inset, and equations occupy distinct regions so labels cannot collide.
footerRows = 3;
geometryHeightInches = 4.15;
footerHeightInches = max(0.80,0.25*footerRows+0.20);
fig = publication_figure(9.0,geometryHeightInches+footerHeightInches);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cBody = [0.70,0.70,0.70];
cOcc = [0.38,0.38,0.38];
cSensor = [0.92,0.55,0.05];
cLos = [0.05,0.05,0.05];
cOccShade = [0.93,0.93,0.93];
cSensorShade = [1.00,0.95,0.86];

% ---------------- Main geometry ----------------
ax = axes(fig,'Units','normalized','Position',[0.055,0.25,0.62,0.69]);
hold(ax,'on');
axis(ax,'equal');
axis(ax,'off');

observer = [-2.60,0.0];
body = [0.95,0.0];
bodyRadius = 0.58;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);
thetaSensor = deg2rad(24);      % schematic example only
thetaKeepout = max(thetaOcc,thetaSensor);
thetaTarget = deg2rad(43);
targetRange = 4.55;
target = observer + targetRange*[cos(thetaTarget),sin(thetaTarget)];

bodyAngle = linspace(0,2*pi,240);
sectorRadius = 2.45;

% Complete configured keepout sector.
keepoutAngles = linspace(-thetaKeepout,thetaKeepout,240);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(keepoutAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(keepoutAngles),observer(2)], ...
    cSensorShade,'EdgeColor','none','HandleVisibility','off');

% Physical occultation cone overlaid inside the keepout sector.
occAngles = linspace(-thetaOcc,thetaOcc,180);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(occAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(occAngles),observer(2)], ...
    cOccShade,'EdgeColor','none','HandleVisibility','off');

% Body, body-center direction, threshold rays, and accepted target LOS.
fill(ax,body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.1);
plot(ax,[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.2);

occEnd = observer + 3.35*[cos(thetaOcc),sin(thetaOcc)];
keepoutEnd = observer + 3.35*[cos(thetaKeepout),sin(thetaKeepout)];
plot(ax,[observer(1),occEnd(1)],[observer(2),occEnd(2)],':', ...
    'Color',cOcc,'LineWidth',1.7);
plot(ax,[observer(1),keepoutEnd(1)],[observer(2),keepoutEnd(2)],'--', ...
    'Color',cSensor,'LineWidth',1.9);
plot(ax,[observer(1),target(1)],[observer(2),target(2)],'-', ...
    'Color',cLos,'LineWidth',2.2);

plot(ax,observer(1),observer(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax,target(1),target(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

% Three nested angular quantities, separated radially to avoid overlap.
draw_angle_arc_2d(ax,observer,0,thetaOcc,0.66,cOcc,1.5);
draw_angle_arc_2d(ax,observer,0,thetaKeepout,1.14,cSensor,1.7);
draw_angle_arc_2d(ax,observer,0,thetaTarget,1.72,cTarget,1.7);

text(ax,observer(1),observer(2)-0.36,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax,target(1)+0.03,target(2)+0.20,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax,body(1)+0.02,body(2)-0.82,'Body b', ...
    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');

text(ax,observer(1)+0.77*cos(thetaOcc/2), ...
    observer(2)+0.77*sin(thetaOcc/2)-0.13,'\theta_{occ,b}', ...
    'Color',cOcc,'FontWeight','bold','FontSize',11);
text(ax,observer(1)+1.28*cos(thetaKeepout/2), ...
    observer(2)+1.28*sin(thetaKeepout/2)+0.07,'\theta_{keepout,b}', ...
    'Color',cSensor,'FontWeight','bold','FontSize',11);
text(ax,observer(1)+1.89*cos(thetaTarget/2), ...
    observer(2)+1.89*sin(thetaTarget/2)+0.08,'\theta_b', ...
    'Color',cTarget,'FontWeight','bold','FontSize',11);

text(ax,-0.70,-0.64,'physical occultation', ...
    'Color',cOcc,'FontSize',10.2,'FontAngle','italic', ...
    'HorizontalAlignment','center');
text(ax,-0.52,0.55,'additional sensor margin', ...
    'Color',cSensor,'FontSize',10.2,'FontAngle','italic', ...
    'HorizontalAlignment','center');

title(ax,'Unified minimum angular-separation geometry', ...
    'FontSize',13.5,'FontWeight','bold');
xlim(ax,[-3.05,2.55]);
ylim(ax,[-1.55,3.18]);

% ---------------- Body-specific threshold inset ----------------
insetAx = axes(fig,'Units','normalized','Position',[0.715,0.36,0.245,0.47]);
axis(insetAx,'off');
rectangle(insetAx,'Position',[0,0,1,1], ...
    'FaceColor',[0.985,0.985,0.985], ...
    'EdgeColor',[0.72,0.72,0.72],'LineWidth',0.9);
text(insetAx,0.50,0.88,'Configured center-separation limits', ...
    'Units','normalized','HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',11.2);
text(insetAx,0.14,0.67,'Earth','Units','normalized', ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',11);
text(insetAx,0.86,0.67,'15^{\circ}','Units','normalized', ...
    'HorizontalAlignment','right','FontName','Times New Roman', ...
    'FontSize',11);
text(insetAx,0.14,0.48,'Moon','Units','normalized', ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',11);
text(insetAx,0.86,0.48,'10^{\circ}','Units','normalized', ...
    'HorizontalAlignment','right','FontName','Times New Roman', ...
    'FontSize',11);
text(insetAx,0.14,0.29,'Sun','Units','normalized', ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',11);
text(insetAx,0.86,0.29,'20^{\circ}','Units','normalized', ...
    'HorizontalAlignment','right','FontName','Times New Roman', ...
    'FontSize',11);
text(insetAx,0.50,0.09,'Same test for all three bodies', ...
    'Units','normalized','HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontAngle','italic','FontSize',9.8);

% ---------------- Compact equation footer ----------------
textAx = axes(fig,'Units','normalized','Position',[0.06,0.035,0.90,0.17]);
axis(textAx,'off');
footerY = [0.82,0.50,0.18];
text(textAx,0.00,footerY(1), ...
    '\theta_{keepout,b}=max(\theta_{occ,b},\theta_{sensor,b}),   b \in {Earth, Moon, Sun}', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',11.7,'Color',cSensor, ...
    'VerticalAlignment','middle');
text(textAx,0.00,footerY(2), ...
    'Admissible LOS: \theta_b>\theta_{occ,b}  and  \theta_b\geq\theta_{sensor,b}', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',10.8,'VerticalAlignment','middle');
text(textAx,0.00,footerY(3), ...
    '\theta_{sensor,b}\rightarrow0  \Rightarrow  \theta_{keepout,b}\rightarrow\theta_{occ,b}', ...
    'Units','normalized','FontName','Times New Roman', ...
    'FontSize',10.8,'VerticalAlignment','middle');

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
% Visibility and keepout constraints intentionally do not appear here.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

footerRows = 1;
geometryHeightInches = 4.00;
footerHeightInches = max(0.58,0.24*footerRows+0.22);
fig = publication_figure(8.9,geometryHeightInches+footerHeightInches);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cProjection = [0.42,0.42,0.42];
cAngle = [0.88,0.43,0.08];

% One internally consistent LOS vector is shown in two orthogonal views.
rho = [3.15,2.10,1.75];
rhoXY = hypot(rho(1),rho(2));
alpha = atan2(rho(2),rho(1));
delta = asin(rho(3)/norm(rho));

% ---------------- (a) Right ascension ----------------
ax1 = axes(fig,'Units','normalized','Position',[0.055,0.20,0.405,0.72]);
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

text(ax1,4.40,-0.08,'x','FontWeight','bold','FontSize',14);
text(ax1,-0.10,3.62,'y','FontWeight','bold','FontSize',14);
text(ax1,0,-0.39,'Observer','Color',cObserver, ...
    'FontWeight','bold','FontSize',12,'HorizontalAlignment','center');
text(ax1,projection(1)+0.06,projection(2)+0.22,'LOS projection', ...
    'Color',cTarget,'FontWeight','bold','FontSize',11);
text(ax1,1.82,1.28,'\rho_{xy}', ...
    'Color',cProjection,'FontWeight','bold','FontSize',12);
text(ax1,1.38*cos(alpha/2),1.38*sin(alpha/2)+0.03,'\alpha', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);

title(ax1,'(a) Right ascension, \alpha', ...
    'FontSize',13.5,'FontWeight','bold');
xlim(ax1,[-0.72,4.65]);
ylim(ax1,[-0.72,3.85]);

% ---------------- (b) Declination ----------------
ax2 = axes(fig,'Units','normalized','Position',[0.535,0.20,0.405,0.72]);
hold(ax2,'on');
axis(ax2,'equal');
axis(ax2,'off');

target = [rhoXY,rho(3)];
quiver(ax2,0,0,4.35,0,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
quiver(ax2,0,0,0,3.25,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
plot(ax2,[0,target(1)],[0,target(2)],'-k','LineWidth',2.2);
plot(ax2,[target(1),target(1)],[0,target(2)],'--', ...
    'Color',cProjection,'LineWidth',1.4);
plot(ax2,[0,target(1)],[0,0],'--', ...
    'Color',cProjection,'LineWidth',1.4);
plot(ax2,0,0,'o','MarkerSize',10,'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax2,target(1),target(2),'o','MarkerSize',9,'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

% Declination is measured above the LOS projection in the x-y plane.
deltaSamples = linspace(0,delta,120);
deltaRadius = 1.22;
plot(ax2,deltaRadius*cos(deltaSamples),deltaRadius*sin(deltaSamples), ...
    '-','Color',cAngle,'LineWidth',2.0);

text(ax2,4.48,-0.08,'\rho_{xy}', ...
    'Color',cProjection,'FontWeight','bold','FontSize',12);
text(ax2,-0.12,3.42,'z','FontWeight','bold','FontSize',14);
text(ax2,0,-0.39,'Observer','Color',cObserver, ...
    'FontWeight','bold','FontSize',12,'HorizontalAlignment','center');
text(ax2,target(1)+0.08,target(2)+0.20,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12);
text(ax2,2.15,1.12,'\rho','FontWeight','bold','FontSize',13);
text(ax2,1.42*cos(delta/2),1.42*sin(delta/2)+0.03,'\delta', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);

title(ax2,'(b) Declination, \delta', ...
    'FontSize',13.5,'FontWeight','bold');
xlim(ax2,[-0.72,4.75]);
ylim(ax2,[-0.72,3.60]);

% Single compact equation row, independent of the geometry axes.
textAx = axes(fig,'Units','normalized','Position',[0.07,0.045,0.86,0.095]);
axis(textAx,'off');
text(textAx,0.50,0.50, ...
    '\rho=r_{tar}-r_{obs}      \alpha=atan2(\rho_y,\rho_x)      \delta=asin(\rho_z/||\rho||)', ...
    'Units','normalized','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',11.8);

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

new_text = text[:start_vis] + visibility + '\n\n\n' + measurement + '\n\n\n' + text[start_helper:]

# Basic repository-side audit: the measurement function must contain no
# visibility/keepout language, while the visibility function must include
# all three bright bodies and the unified threshold.
vis_region = new_text[new_text.index('function outputs = create_visibility_keepout_figure'):new_text.index('function outputs = create_measurement_model_figure')]
meas_region = new_text[new_text.index('function outputs = create_measurement_model_figure'):new_text.index('function [fig,axesHandles,textAx] = schematic_figure_layout')]

assert 'theta_{keepout,b}=max' in vis_region
assert 'Earth, Moon, Sun' in vis_region
assert 'visibility' not in meas_region.lower()
assert 'keepout' not in meas_region.lower()
assert 'theta_' not in meas_region

path.write_text(new_text)
print('Updated visibility and measurement schematics.')
