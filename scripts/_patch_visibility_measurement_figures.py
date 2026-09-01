from pathlib import Path
import re

path = Path('scripts/plot_study_definition_figures.m')
text = path.read_text()

visibility = r'''function outputs = create_visibility_keepout_figure(inspectFigure)
% Illustrate the unified visibility framework used by calc_visibility.
% For each body b in {Earth, Moon, Sun}, visibility is enforced through
% one angular-separation framework. The effective keepout angle is
% theta_keepout,b = max(theta_occ,b, theta_sensor,b).

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

fig = publication_figure(8.8,4.6);
layout = tiledlayout(fig,1,2,'TileSpacing','loose','Padding','compact');

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cBody = [0.68,0.68,0.68];
cOcc = [0.34,0.34,0.34];
cSensor = [0.92,0.55,0.05];
cKeepout = [0.72,0.13,0.12];
cShade = [1.00,0.95,0.87];

% ---- (a) Unified angular-separation geometry ----
ax = nexttile(layout,1);
hold(ax,'on');
axis(ax,'equal');
axis(ax,'off');

observer = [-2.35,0.00];
body = [0.25,0.00];
bodyRadius = 0.52;
thetaOcc = deg2rad(14);
thetaSensor = deg2rad(26);
thetaKeepout = max(thetaOcc,thetaSensor);
thetaBody = deg2rad(43);
target = observer + 4.40*[cos(thetaBody),sin(thetaBody)];

% Keepout sector.
sectorAngle = linspace(-thetaKeepout,thetaKeepout,180);
sectorRadius = 2.15;
patch(ax, ...
    [observer(1),observer(1)+sectorRadius*cos(sectorAngle),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(sectorAngle),observer(2)], ...
    cShade,'EdgeColor','none','HandleVisibility','off');

% Body and centerline.
bodyAngle = linspace(0,2*pi,240);
fill(ax,body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.0);
plot(ax,[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.2);

% Occultation and sensor threshold rays.
occEnd = observer + 3.10*[cos(thetaOcc),sin(thetaOcc)];
sensorEnd = observer + 3.10*[cos(thetaSensor),sin(thetaSensor)];
plot(ax,[observer(1),occEnd(1)],[observer(2),occEnd(2)],':', ...
    'Color',cOcc,'LineWidth',1.5);
plot(ax,[observer(1),sensorEnd(1)],[observer(2),sensorEnd(2)],'--', ...
    'Color',cSensor,'LineWidth',1.7);

% Accepted target LOS.
plot(ax,[observer(1),target(1)],[observer(2),target(2)],'-k', ...
    'LineWidth',2.0);
plot(ax,observer(1),observer(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax,target(1),target(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

% Angles.
draw_angle_arc_2d(ax,observer,0,thetaOcc,0.64,cOcc,1.4);
draw_angle_arc_2d(ax,observer,thetaOcc,thetaSensor,0.93,cSensor,1.5);
draw_angle_arc_2d(ax,observer,0,thetaBody,1.30,cKeepout,1.6);

% Sparse labels.
text(ax,observer(1),observer(2)-0.36,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax,target(1)+0.04,target(2)+0.24,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax,body(1),body(2)-0.10,'Body b', ...
    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');
text(ax,0.10,-0.82,'b \in {Earth, Moon, Sun}', ...
    'FontWeight','bold','FontSize',10.5,'HorizontalAlignment','center');

text(ax,observer(1)+0.55*cos(thetaOcc/2), ...
    observer(2)+0.55*sin(thetaOcc/2)-0.14, ...
    '\theta_{occ,b}','Color',cOcc,'FontWeight','bold','FontSize',11);
text(ax,observer(1)+1.00*cos((thetaOcc+thetaSensor)/2), ...
    observer(2)+1.00*sin((thetaOcc+thetaSensor)/2)+0.05, ...
    '\theta_{sensor,b}','Color',cSensor,'FontWeight','bold','FontSize',11);
text(ax,observer(1)+1.38*cos(thetaBody/2), ...
    observer(2)+1.38*sin(thetaBody/2)+0.06, ...
    '\theta_b','Color',cKeepout,'FontWeight','bold','FontSize',11);

text(ax,-2.72,-1.48, ...
    '\theta_{keepout,b} = max(\theta_{occ,b},\theta_{sensor,b})', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',10.8, ...
    'BackgroundColor','w','EdgeColor',[0.78,0.78,0.78], ...
    'Margin',5);
text(ax,-2.72,-1.88, ...
    'Visible: \theta_b > \theta_{occ,b} and \theta_b \geq \theta_{sensor,b}', ...
    'FontWeight','bold','FontSize',9.8);

text(ax,-2.72,1.74, ...
    'Sensor thresholds: Earth 15^o   Moon 10^o   Sun 20^o', ...
    'FontWeight','bold','FontSize',9.8);

title(ax,'(a) Unified minimum angular separation', ...
    'FontSize',13,'FontWeight','bold');
xlim(ax,[-2.95,3.05]);
ylim(ax,[-2.12,2.08]);

% ---- (b) Exclusion-angle limit becomes physical occlusion ----
ax = nexttile(layout,2);
hold(ax,'on');
axis(ax,'equal');
axis(ax,'off');

observer = [-2.35,0.00];
body = [0.25,0.00];
bodyRadius = 0.52;
thetaOcc = deg2rad(17);
thetaSensor = 0;
thetaKeepout = max(thetaOcc,thetaSensor);
thetaBody = deg2rad(38);
target = observer + 4.35*[cos(thetaBody),sin(thetaBody)];

sectorAngle = linspace(-thetaKeepout,thetaKeepout,180);
sectorRadius = 2.15;
patch(ax, ...
    [observer(1),observer(1)+sectorRadius*cos(sectorAngle),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(sectorAngle),observer(2)], ...
    [0.94,0.94,0.94],'EdgeColor','none','HandleVisibility','off');

fill(ax,body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.0);
plot(ax,[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.2);

boundaryEnd = observer + 3.15*[cos(thetaOcc),sin(thetaOcc)];
plot(ax,[observer(1),boundaryEnd(1)], ...
    [observer(2),boundaryEnd(2)],'--', ...
    'Color',cKeepout,'LineWidth',1.7);
plot(ax,[observer(1),target(1)],[observer(2),target(2)],'-k', ...
    'LineWidth',2.0);

plot(ax,observer(1),observer(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax,target(1),target(2),'o','MarkerSize',9, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

draw_angle_arc_2d(ax,observer,0,thetaOcc,0.86,cKeepout,1.6);
draw_angle_arc_2d(ax,observer,0,thetaBody,1.31,cKeepout,1.6);

text(ax,observer(1),observer(2)-0.36,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax,target(1)+0.05,target(2)+0.24,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','center');
text(ax,body(1),body(2)-0.10,'Body b', ...
    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');

text(ax,observer(1)+0.78*cos(thetaOcc/2), ...
    observer(2)+0.78*sin(thetaOcc/2)+0.08, ...
    '\theta_{keepout,b}=\theta_{occ,b}', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',10.8);
text(ax,observer(1)+1.38*cos(thetaBody/2), ...
    observer(2)+1.38*sin(thetaBody/2)+0.06, ...
    '\theta_b','Color',cKeepout,'FontWeight','bold','FontSize',11);

text(ax,-2.72,-1.38, ...
    '\theta_{sensor,b} \rightarrow 0', ...
    'Color',cSensor,'FontWeight','bold','FontSize',11.5, ...
    'BackgroundColor','w','EdgeColor',[0.78,0.78,0.78], ...
    'Margin',5);
text(ax,-2.72,-1.78, ...
    '\theta_{keepout,b} \rightarrow \theta_{occ,b}', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',11.5);
text(ax,-2.72,-2.08, ...
    'The exclusion constraint therefore reduces to physical occlusion.', ...
    'FontWeight','bold','FontSize',9.8);

text(ax,-2.72,1.74, ...
    'Same formulation for Earth, Moon, and Sun', ...
    'FontWeight','bold','FontSize',10.2);

title(ax,'(b) Zero-threshold limit','FontSize',13,'FontWeight','bold');
xlim(ax,[-2.95,3.05]);
ylim(ax,[-2.30,2.08]);

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
% Illustrate the implemented angles-only measurement model:
% rho = r_target-r_observer, alpha = atan2(rho_y,rho_x), and
% delta = asin(rho_z/|rho|). Unified visibility screening is applied first.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

fig = publication_figure(7.6,5.7);
ax = axes(fig);
hold(ax,'on');
axis(ax,'equal');
axis(ax,'vis3d');
axis(ax,'off');
view(ax,-35,24);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cProjection = [0.38,0.38,0.38];
cPlane = [0.90,0.95,0.98];
cAngle = [0.88,0.43,0.08];
cKeepout = [0.72,0.13,0.12];

observer = [0,0,0];
target = [3.05,2.05,1.75];
projection = [target(1),target(2),0];
rho = target-observer;
rhoXY = norm(rho(1:2));
alpha = atan2(rho(2),rho(1));
delta = asin(rho(3)/norm(rho));

% Reference x-y plane.
planeX = [-1.8,4.1];
planeY = [-1.7,3.2];
[planeXX,planeYY] = meshgrid(planeX,planeY);
surf(ax,planeXX,planeYY,zeros(size(planeXX)), ...
    'FaceColor',cPlane,'EdgeColor',[0.72,0.80,0.84], ...
    'LineWidth',0.7);

% Cartesian axes.
quiver3(ax,0,0,0,3.90,0,0,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.10);
quiver3(ax,0,0,0,0,3.45,0,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.10);
quiver3(ax,0,0,0,0,0,3.15,0,'Color','k','LineWidth',1.8, ...
    'MaxHeadSize',0.10);

% LOS and its x-y projection.
plot3(ax,[0,target(1)],[0,target(2)],[0,target(3)],'-k', ...
    'LineWidth',2.3);
plot3(ax,[0,projection(1)],[0,projection(2)],[0,0],'--', ...
    'Color',cProjection,'LineWidth',1.5);
plot3(ax,[projection(1),target(1)], ...
    [projection(2),target(2)],[0,target(3)],'--', ...
    'Color',cProjection,'LineWidth',1.2);

plot3(ax,observer(1),observer(2),observer(3),'o', ...
    'MarkerSize',12,'MarkerFaceColor',cObserver,'MarkerEdgeColor','k', ...
    'LineWidth',1.0);
plot3(ax,target(1),target(2),target(3),'o', ...
    'MarkerSize',12,'MarkerFaceColor',cTarget,'MarkerEdgeColor','k', ...
    'LineWidth',1.0);
plot3(ax,projection(1),projection(2),projection(3),'o', ...
    'MarkerSize',5,'MarkerFaceColor','w','MarkerEdgeColor',cProjection, ...
    'LineWidth',1.0);

% Right ascension in the x-y plane.
alphaSamples = linspace(0,alpha,120);
alphaRadius = 1.12;
plot3(ax,alphaRadius*cos(alphaSamples), ...
    alphaRadius*sin(alphaSamples),zeros(size(alphaSamples)), ...
    '-','Color',cAngle,'LineWidth',2.0);

% Declination in the vertical plane containing rho.
uXY = rho(1:2)/rhoXY;
deltaSamples = linspace(0,delta,120);
deltaRadius = 1.28;
deltaArc = [ ...
    deltaRadius*cos(deltaSamples(:))*uXY(1), ...
    deltaRadius*cos(deltaSamples(:))*uXY(2), ...
    deltaRadius*sin(deltaSamples(:))];
plot3(ax,deltaArc(:,1),deltaArc(:,2),deltaArc(:,3), ...
    '-','Color',cAngle,'LineWidth',2.0);

% Geometry labels.
text(ax,4.04,0,0,'x','FontWeight','bold','FontSize',14);
text(ax,0,3.62,0,'y','FontWeight','bold','FontSize',14);
text(ax,0,0,3.30,'z','FontWeight','bold','FontSize',14);
text(ax,-1.55,-1.40,0.03,'x-y plane', ...
    'FontWeight','bold','FontSize',12);

text(ax,-0.18,-0.18,0.28,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','right');
text(ax,target(1)+0.18,target(2)+0.10,target(3)+0.16,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',12);
text(ax,1.55,1.08,1.08,'\rho', ...
    'FontWeight','bold','FontSize',13);
text(ax,1.42,0.95,0.05,'\rho_{xy}', ...
    'Color',cProjection,'FontWeight','bold','FontSize',11);

alphaMid = alpha/2;
text(ax,1.28*cos(alphaMid),1.28*sin(alphaMid),0.06,'\alpha', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);
deltaMid = delta/2;
deltaLabel = [ ...
    1.48*cos(deltaMid)*uXY(1), ...
    1.48*cos(deltaMid)*uXY(2), ...
    1.48*sin(deltaMid)];
text(ax,deltaLabel(1),deltaLabel(2),deltaLabel(3),'\delta', ...
    'Color',cAngle,'FontWeight','bold','FontSize',15);

% Compact equations placed away from the geometry.
formulaText = { ...
    '\rho = r_{tar}-r_{obs}', ...
    '\alpha = atan2(\rho_y,\rho_x)', ...
    '\delta = asin(\rho_z/||\rho||)'};
text(ax,-1.62,2.92,2.58,formulaText, ...
    'FontWeight','bold','FontSize',11.2, ...
    'BackgroundColor','w','EdgeColor',[0.76,0.76,0.76], ...
    'Margin',6);

text(ax,-1.62,-1.52,2.48, ...
    'Visibility gate first: \theta_b \geq \theta_{keepout,b},  b \in {Earth, Moon, Sun}', ...
    'Color',cKeepout,'FontWeight','bold','FontSize',10.2, ...
    'BackgroundColor','w','EdgeColor',[0.80,0.80,0.80], ...
    'Margin',5);

title(ax,'Angles-only relative line-of-sight measurement', ...
    'FontSize',14,'FontWeight','bold');
xlim(ax,[-1.9,4.25]);
ylim(ax,[-1.8,3.65]);
zlim(ax,[-0.1,3.35]);
set(ax,'FontName','Times New Roman','FontSize',13,'FontWeight','bold');
set(findall(fig,'Type','text'),'FontName','Times New Roman');
ax.Position = [0.07,0.06,0.87,0.84];

figureFile = fullfile(outputDir,'measurement_model_radec_geometry.eps');
inspect_before_export(fig,inspectFigure,'RA/Dec measurement geometry');
export_publication_eps(fig,figureFile);
close(fig);

outputs = struct();
outputs.figure = string(figureFile);
outputs.measurementType = "ANGLES_ONLY";
fprintf('Saved RA/Dec measurement geometry to:\n  %s\n',figureFile);
end'''

pattern_visibility = re.compile(
    r'function outputs = create_visibility_keepout_figure\(inspectFigure\).*?(?=\nfunction outputs = create_measurement_model_figure\(inspectFigure\))',
    re.S)
pattern_measurement = re.compile(
    r'function outputs = create_measurement_model_figure\(inspectFigure\).*?(?=\nfunction draw_angle_arc_2d\()',
    re.S)

text, n1 = pattern_visibility.subn(visibility + '\n\n', text, count=1)
if n1 != 1:
    raise SystemExit(f'Expected one visibility function replacement, got {n1}.')

text, n2 = pattern_measurement.subn(measurement + '\n\n', text, count=1)
if n2 != 1:
    raise SystemExit(f'Expected one measurement function replacement, got {n2}.')

# Guard the requested 16-orbit presentation.
text = text.replace('numPairOrbitsPerFamily = 10;', 'numPairOrbitsPerFamily = 16;')
text = text.replace('numDroOrbits = 10;', 'numDroOrbits = 16;')

path.write_text(text)

# Lightweight static checks.
check = path.read_text()
required = [
    'theta_{keepout,b} = max',
    'b \\in {Earth, Moon, Sun}',
    'theta_{sensor,b} \\rightarrow 0',
    'measurement_model_radec_geometry.eps',
    'visibility_keepout_geometry.eps',
    'numPairOrbitsPerFamily = 16;',
    'numDroOrbits = 16;',
]
for item in required:
    if item not in check:
        raise SystemExit(f'Missing expected text: {item}')

# Self-cleanup so the final commit contains only the production MATLAB change.
Path('scripts/_patch_visibility_measurement_figures.py').unlink(missing_ok=True)
Path('.github/workflows/one-time-visibility-measurement-polish.yml').unlink(missing_ok=True)
