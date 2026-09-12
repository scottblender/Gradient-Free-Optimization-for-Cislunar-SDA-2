function outputs = plot_visibility_keepout_geometry(inspectFigure,outputDirectory)
%PLOT_VISIBILITY_KEEPOUT_GEOMETRY Render only the visibility/keepout schematic.
%
% outputs = plot_visibility_keepout_geometry(inspectFigure,outputDirectory)
%
% This standalone renderer reproduces the manuscript occultation/keepout
% geometry without regenerating the other study-definition figures. The
% target label is kept slightly farther left and uses a smaller font so it
% remains fully inside the exported EPS/PNG canvas.

if nargin < 1 || isempty(inspectFigure), inspectFigure = false; end
validateattributes(inspectFigure,{'logical','numeric'},{'scalar'});
inspectFigure = logical(inspectFigure);

paths = setup_project();
if nargin < 2 || strlength(string(outputDirectory)) == 0
    outputDirectory = fullfile(paths.results,'study_definition_figures');
end
outputDirectory = char(string(outputDirectory));
if ~isfolder(outputDirectory), mkdir(outputDirectory); end

style = reviewer2_paper_style();
fig = figure( ...
    'Color','w', ...
    'Units','inches', ...
    'Position',[1,1,style.visibilityFigureWidth,style.visibilityFigureHeight], ...
    'PaperUnits','inches', ...
    'PaperPosition',[0,0,style.visibilityFigureWidth,style.visibilityFigureHeight], ...
    'PaperSize',[style.visibilityFigureWidth,style.visibilityFigureHeight], ...
    'PaperPositionMode','manual', ...
    'Renderer','painters', ...
    'InvertHardcopy','off');
cleanup = onCleanup(@() close_if_valid(fig)); %#ok<NASGU>

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cBody = [0.70,0.70,0.70];
cOcc = [0.38,0.38,0.38];
cExclusion = [0.92,0.55,0.05];
cLos = [0.05,0.05,0.05];
cOccShade = [0.82,0.82,0.82];
cExclusionShade = [0.98,0.89,0.62];

observer = [-2.20,0.00];
body = [1.05,0.00];
bodyRadius = 0.66;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);
thetaExclusion = thetaOcc + deg2rad(8);
thetaKeepout = max(thetaOcc,thetaExclusion);
thetaB = thetaKeepout + deg2rad(13);
targetRange = 4.75;
target = observer + targetRange*[cos(thetaB),sin(thetaB)];

ax = axes(fig,'Units','normalized','Position',[0.05,0.06,0.90,0.88]);
hold(ax,'on');
box(ax,'off');
grid(ax,'off');
axis(ax,'equal');
axis(ax,'off');

bodyAngle = linspace(0,2*pi,240);
sectorRadius = 3.45;

occAngles = linspace(-thetaOcc,thetaOcc,220);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(occAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(occAngles),observer(2)], ...
    cOccShade,'EdgeColor','none','HandleVisibility','off');

upperMargin = linspace(thetaOcc,thetaKeepout,120);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(upperMargin),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(upperMargin),observer(2)], ...
    cExclusionShade,'EdgeColor','none','HandleVisibility','off');
lowerMargin = linspace(-thetaKeepout,-thetaOcc,120);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(lowerMargin),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(lowerMargin),observer(2)], ...
    cExclusionShade,'EdgeColor','none','HandleVisibility','off');

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
    'Color',cExclusion,'LineWidth',2.1);
plot(ax,[observer(1),target(1)],[observer(2),target(2)],'-', ...
    'Color',cLos,'LineWidth',2.5);

plot(ax,observer(1),observer(2),'o','MarkerSize',11, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax,target(1),target(2),'o','MarkerSize',11, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

draw_angle_arc_2d(ax,observer,0,thetaOcc,0.95,cOcc,1.9);
draw_angle_arc_2d(ax,observer,0,thetaKeepout,1.48,cExclusion,2.1);
draw_angle_arc_2d(ax,observer,0,thetaB,2.10,cTarget,2.1);

text(ax,observer(1)-0.02,observer(2)-0.46,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','center');
% Keep the target label clear of the right export boundary.
text(ax,target(1)+0.12,target(2)-0.02,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',15, ...
    'HorizontalAlignment','left');
text(ax,body(1),body(2)-1.16,'Body b', ...
    'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','center','BackgroundColor','w', ...
    'Margin',0.8);

occArcPoint = observer + ...
    0.95*[cos(0.5*thetaOcc),sin(0.5*thetaOcc)];
keepoutArcPoint = observer + ...
    1.48*[cos(0.5*thetaKeepout),sin(0.5*thetaKeepout)];
targetArcPoint = observer + ...
    2.10*[cos(0.5*thetaB),sin(0.5*thetaB)];

occLabel = occArcPoint + [0.55,-0.42];
keepoutLabel = keepoutArcPoint + [-0.55,0.55];
targetAngleLabel = targetArcPoint + [0.35,0.42];

occLeaderDirection = occArcPoint-occLabel;
occLeaderPoint = occArcPoint- ...
    0.10*occLeaderDirection/norm(occLeaderDirection);

draw_text_callout(ax,occLabel,occLeaderPoint, ...
    '\theta_{occ,b}',cOcc,16);
draw_text_callout(ax,keepoutLabel,keepoutArcPoint, ...
    '\theta_{keepout,b}',cExclusion,16);
draw_text_callout(ax,targetAngleLabel,targetArcPoint, ...
    '\theta_b',cTarget,16);

ptOccRegion = observer + ...
    2.45*[cos(-0.55*thetaOcc),sin(-0.55*thetaOcc)];
occCallout = observer + [1.65,-1.28];
text(ax,occCallout(1),occCallout(2), ...
    {'physical';'occultation'}, ...
    'Color',cOcc,'FontSize',15,'FontAngle','italic', ...
    'FontWeight','bold','HorizontalAlignment','center', ...
    'VerticalAlignment','middle');
occArrowStart = occCallout + [0,0.28];
draw_leader_arrow(ax,occArrowStart,ptOccRegion,cOcc);

ptMargin = observer + ...
    2.70*[cos(0.5*(thetaOcc+thetaKeepout)), ...
    sin(0.5*(thetaOcc+thetaKeepout))];
marginCallout = observer + [2.92,2.08];
text(ax,marginCallout(1),marginCallout(2), ...
    {'effective';'exclusion margin'}, ...
    'Color',cExclusion,'FontSize',15,'FontAngle','italic', ...
    'FontWeight','bold','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','BackgroundColor','w', ...
    'Margin',0.8);
marginArrowStart = marginCallout + [0,-0.30];
draw_leader_arrow(ax,marginArrowStart,ptMargin,cExclusion);

xlim(ax,[-3.20,3.25]);
ylim(ax,[-2.10,3.50]);
set(findall(fig,'Type','text'),'FontName',style.fontName);

if inspectFigure
    figure(fig);
    drawnow;
    fprintf('Previewing the unified visibility / keepout geometry for 5 seconds before export.\n');
    pause(5);
end

figureFile = fullfile(outputDirectory,'visibility_keepout_geometry.eps');
export_manuscript_figure(fig,figureFile);

outputs = struct();
outputs.figure = string(figureFile);
outputs.thetaExclusion_deg = struct('earth',15,'moon',10,'sun',20);
fprintf('Saved visibility / keepout geometry to:\n  %s\n',figureFile);
end


function draw_angle_arc_2d(ax,origin,startAngle,endAngle,radius,color,lineWidth)
angle = linspace(startAngle,endAngle,100);
plot(ax,origin(1)+radius*cos(angle),origin(2)+radius*sin(angle), ...
    '-','Color',color,'LineWidth',lineWidth);
end


function draw_text_callout(ax,labelPosition,targetPosition,labelText,labelColor,fontSize)
fontSize = max(fontSize,12);
direction = targetPosition-labelPosition;
distance = norm(direction);
if distance > 0
    leaderOffset = min(0.22,0.30*distance);
    arrowStart = labelPosition + leaderOffset*direction/distance;
else
    arrowStart = labelPosition;
end
text(ax,labelPosition(1),labelPosition(2),labelText, ...
    'Color',labelColor,'FontWeight','bold','FontSize',fontSize, ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'BackgroundColor','w','Margin',0.8);
draw_leader_arrow(ax,arrowStart,targetPosition,labelColor);
end


function draw_leader_arrow(ax,startPosition,targetPosition,lineColor)
delta = targetPosition-startPosition;
distance = norm(delta);
if distance <= eps, return; end
direction = delta/distance;
normal = [-direction(2),direction(1)];
headLength = min(0.13,0.20*distance);
headHalfWidth = 0.48*headLength;
headBase = targetPosition-headLength*direction;
plot(ax,[startPosition(1),headBase(1)], ...
    [startPosition(2),headBase(2)],'-', ...
    'Color',lineColor,'LineWidth',1.55, ...
    'Clipping','off','HandleVisibility','off');
headX = [targetPosition(1), ...
    headBase(1)+headHalfWidth*normal(1), ...
    headBase(1)-headHalfWidth*normal(1)];
headY = [targetPosition(2), ...
    headBase(2)+headHalfWidth*normal(2), ...
    headBase(2)-headHalfWidth*normal(2)];
patch(ax,headX,headY,lineColor, ...
    'EdgeColor',lineColor,'LineWidth',0.8, ...
    'Clipping','off','HandleVisibility','off');
end


function close_if_valid(fig)
if isgraphics(fig), close(fig); end
end
