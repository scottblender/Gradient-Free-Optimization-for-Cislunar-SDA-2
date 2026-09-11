function outputs = plot_visibility_keepout_manuscript(inspectFigure,outputDir)
%PLOT_VISIBILITY_KEEPOUT_MANUSCRIPT Generate the final keep-out schematic.
% Styling and geometry sizing are completed before export.

if nargin<1 || isempty(inspectFigure), inspectFigure = false; end
paths = setup_project();
if nargin<2 || strlength(string(outputDir))==0
    outputDir = fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
outputDir = char(string(outputDir));
if ~isfolder(outputDir), mkdir(outputDir); end

style = reviewer2_paper_style();
fig = manuscript_figure(style.visibilityFigureWidth,style.visibilityFigureHeight,style);
ax = axes(fig,'Units','normalized','Position',style.visibilityPlotPosition);
hold(ax,'on'); box(ax,'off'); grid(ax,'off'); axis(ax,'equal'); axis(ax,'off');

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

fill(ax,body(1)+bodyRadius*cos(bodyAngle),body(2)+bodyRadius*sin(bodyAngle), ...
    cBody,'EdgeColor','k','LineWidth',1.2);
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

draw_angle_arc(ax,observer,0,thetaOcc,0.95,cOcc,1.9);
draw_angle_arc(ax,observer,0,thetaKeepout,1.48,cExclusion,2.1);
draw_angle_arc(ax,observer,0,thetaB,2.10,cTarget,2.1);

text(ax,observer(1)-0.02,observer(2)-0.46,'Observer', ...
    'Color',cObserver,'HorizontalAlignment','center');
text(ax,target(1)+0.18,target(2)-0.02,'Target', ...
    'Color',cTarget,'HorizontalAlignment','left');
text(ax,body(1),body(2)-1.16,'Body b', ...
    'HorizontalAlignment','center','BackgroundColor','w','Margin',0.8);

occArcPoint = observer + 0.95*[cos(0.5*thetaOcc),sin(0.5*thetaOcc)];
keepoutArcPoint = observer + 1.48*[cos(0.5*thetaKeepout),sin(0.5*thetaKeepout)];
targetArcPoint = observer + 2.10*[cos(0.5*thetaB),sin(0.5*thetaB)];
occLabel = occArcPoint + [0.55,-0.42];
keepoutLabel = keepoutArcPoint + [-0.55,0.55];
targetAngleLabel = targetArcPoint + [0.35,0.42];
occLeaderDirection = occArcPoint-occLabel;
occLeaderPoint = occArcPoint-0.10*occLeaderDirection/norm(occLeaderDirection);
draw_text_callout(ax,occLabel,occLeaderPoint,'\theta_{occ,b}',cOcc,style);
draw_text_callout(ax,keepoutLabel,keepoutArcPoint,'\theta_{keepout,b}',cExclusion,style);
draw_text_callout(ax,targetAngleLabel,targetArcPoint,'\theta_b',cTarget,style);

ptOccRegion = observer + 2.45*[cos(-0.55*thetaOcc),sin(-0.55*thetaOcc)];
occCallout = observer + [1.65,-1.28];
text(ax,occCallout(1),occCallout(2),{'physical';'occultation'}, ...
    'Color',cOcc,'FontAngle','italic','HorizontalAlignment','center', ...
    'VerticalAlignment','middle');
draw_leader_arrow(ax,occCallout+[0,0.28],ptOccRegion,cOcc);

ptMargin = observer + 2.70*[cos(0.5*(thetaOcc+thetaKeepout)), ...
    sin(0.5*(thetaOcc+thetaKeepout))];
marginCallout = observer + [2.92,2.08];
text(ax,marginCallout(1),marginCallout(2),{'effective';'exclusion margin'}, ...
    'Color',cExclusion,'FontAngle','italic','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','BackgroundColor','w','Margin',0.8);
draw_leader_arrow(ax,marginCallout+[0,-0.30],ptMargin,cExclusion);

% These tighter limits are the important sizing change: the physical
% geometry occupies more of the EPS instead of merely increasing whitespace.
xlim(ax,style.visibilityXLim);
ylim(ax,style.visibilityYLim);
ax.Position = style.visibilityPlotPosition;
drawnow;

if inspectFigure
    figure(fig); drawnow;
    fprintf('Previewing unified visibility / keepout geometry for 5 seconds before export.\n');
    pause(5);
end

figureFile = fullfile(outputDir,'visibility_keepout_geometry.eps');
export_manuscript_figure(fig,figureFile);
close(fig);
outputs = struct('figure',string(figureFile), ...
    'thetaExclusion_deg',struct('earth',15,'moon',10,'sun',20));
fprintf('Saved visibility / keepout geometry to:\n  %s\n',figureFile);
end


function fig = manuscript_figure(widthIn,heightIn,style)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperPosition',[0 0 widthIn heightIn], ...
    'PaperSize',[widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
set(fig,'DefaultAxesFontName',style.fontName, ...
    'DefaultAxesFontSize',style.fontSize, ...
    'DefaultAxesFontWeight',style.fontWeight, ...
    'DefaultTextFontName',style.fontName, ...
    'DefaultTextFontSize',style.fontSize, ...
    'DefaultTextFontWeight',style.fontWeight);
end


function draw_angle_arc(ax,origin,startAngle,endAngle,radius,color,lineWidth)
a = linspace(startAngle,endAngle,100);
plot(ax,origin(1)+radius*cos(a),origin(2)+radius*sin(a),'-', ...
    'Color',color,'LineWidth',lineWidth,'HandleVisibility','off');
end


function draw_text_callout(ax,labelPosition,targetPosition,labelText,labelColor,style)
direction = targetPosition-labelPosition;
distance = norm(direction);
if distance>0
    arrowStart = labelPosition + min(0.22,0.30*distance)*direction/distance;
else
    arrowStart = labelPosition;
end
text(ax,labelPosition(1),labelPosition(2),labelText, ...
    'Color',labelColor,'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'BackgroundColor','w','Margin',0.8,'FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight',style.fontWeight);
draw_leader_arrow(ax,arrowStart,targetPosition,labelColor);
end


function draw_leader_arrow(ax,startPosition,targetPosition,lineColor)
delta = targetPosition-startPosition;
distance = norm(delta);
if distance<=eps, return; end
direction = delta/distance;
normal = [-direction(2),direction(1)];
headLength = min(0.13,0.20*distance);
headHalfWidth = 0.48*headLength;
headBase = targetPosition-headLength*direction;
plot(ax,[startPosition(1),headBase(1)],[startPosition(2),headBase(2)],'-', ...
    'Color',lineColor,'LineWidth',1.55,'Clipping','off','HandleVisibility','off');
headX = [targetPosition(1),headBase(1)+headHalfWidth*normal(1), ...
    headBase(1)-headHalfWidth*normal(1)];
headY = [targetPosition(2),headBase(2)+headHalfWidth*normal(2), ...
    headBase(2)-headHalfWidth*normal(2)];
patch(ax,headX,headY,lineColor,'EdgeColor',lineColor,'LineWidth',0.8, ...
    'Clipping','off','HandleVisibility','off');
end
