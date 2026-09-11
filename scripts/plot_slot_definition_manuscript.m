function outputs = plot_slot_definition_manuscript(inspectFigure,outputDir)
%PLOT_SLOT_DEFINITION_MANUSCRIPT Final 50-slot definition figures.
% All styling, marker fill, tick selection, and legend placement occur before
% export_manuscript_figure is called.

if nargin<1 || isempty(inspectFigure), inspectFigure = false; end
paths = setup_project();
if nargin<2 || strlength(string(outputDir))==0
    outputDir = fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
outputDir = char(string(outputDir));
if ~isfolder(outputDir), mkdir(outputDir); end
style = reviewer2_paper_style();

catalog = load(paths.catalog,'T');
T = catalog.T;
family = string(T.orbitFamily);
orbitIndex = find(family=="NNRHL1",1,'first');
assert(~isempty(orbitIndex),'No representative northern NRHO L1 orbit was found.');
periodAll = T.('Period (TU) ');
period = periodAll(orbitIndex);
rawTime = T.time{orbitIndex};
rawState = T.state{orbitIndex};
[uniqueTime,uniqueIndex] = unique(rawTime);
uniqueState = rawState(uniqueIndex,:);
interpolant = griddedInterpolant(uniqueTime,uniqueState,'spline');

numSlots = 50;
deltaTime = period/numSlots;
slotNumber = (1:numSlots).';
slotTime = (slotNumber-1)*deltaTime;
slotState = interpolant(slotTime);
assert(slotTime(1)==0);
assert(abs(slotTime(end)-49*period/50)<=10*eps(period));
assert(all(diff(slotTime)>0) && all(slotTime<period));

nextPosition = [slotState(2:end,1:3);slotState(1,1:3)];
adjacentChord_km = vecnorm(nextPosition-slotState(:,1:3),2,2)*384400;
selectedSlot = 17;
nextSlot = selectedSlot+1;
selectedColor = [0.85,0.25,0.20];
nextColor = [0.20,0.50,0.80];
orbitColor = [0.27,0.31,0.86];
neutralColor = [0.25,0.25,0.25];
mu = 1.215058560962404E-2;
LU = 384400;

%% Geometry panel
figGeometry = manuscript_figure(style.geometryFigureWidth,style.geometryFigureHeight,style);
ax = axes(figGeometry,'Units','normalized','Position',style.geometryPlotPosition);
ax.PositionConstraint = 'innerposition';
prepare_geometry_axes(ax,style);
plotStep = max(1,round(size(rawState,1)/500));
hOrbit = plot3(ax,rawState(1:plotStep:end,1),rawState(1:plotStep:end,2), ...
    rawState(1:plotStep:end,3),'-','Color',orbitColor,'LineWidth',2.5);
hSlots = plot3(ax,slotState(:,1),slotState(:,2),slotState(:,3),'o', ...
    'MarkerSize',5,'MarkerFaceColor',style.slotCandidateFillColor, ...
    'MarkerEdgeColor',neutralColor,'LineWidth',1.0);
hSelected = plot3(ax,slotState(selectedSlot,1),slotState(selectedSlot,2), ...
    slotState(selectedSlot,3),'o','MarkerSize',9,'MarkerFaceColor',selectedColor, ...
    'MarkerEdgeColor','k','LineWidth',1.2);
hNext = plot3(ax,slotState(nextSlot,1),slotState(nextSlot,2), ...
    slotState(nextSlot,3),'s','MarkerSize',9,'MarkerFaceColor',nextColor, ...
    'MarkerEdgeColor','k','LineWidth',1.2);
hMoon = draw_moon(ax,mu,LU);
axis(ax,'tight');
xlim(ax,pad_limits(ax.XLim,style.geometryXPadding));
ylim(ax,pad_limits(ax.YLim,style.geometryYPadding));
zlim(ax,pad_limits(ax.ZLim,style.geometryZPadding));
axis(ax,'vis3d');
simplify_geometry_ticks(ax);
lgd = legend(ax,[hOrbit,hSlots,hSelected,hNext,hMoon], ...
    {'Orbit','Candidate slots','Slot j','Slot j+1','Moon'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',3,'Box','off');
style_legend(lgd,style,[16 9]);
place_geometry_legend(ax,lgd,style.geometryPlotPosition,style);
geometryFile = fullfile(outputDir,'slot_geometry_equal_time.eps');
preview_if_requested(figGeometry,inspectFigure,'equal-time slot geometry');
export_manuscript_figure(figGeometry,geometryFile);
close(figGeometry);

%% Phase-grid panel
figPhase = manuscript_figure(style.slotPhaseFigureWidth,style.slotPhaseFigureHeight,style);
ax = axes(figPhase,'Units','normalized','Position',[0.12,0.19,0.80,0.56]);
hold(ax,'on'); box(ax,'off'); grid(ax,'off');
phase = slotTime/period;
plot(ax,[0,1],[0,0],'-','Color',0.65*[1,1,1],'LineWidth',1.5);
hCandidate = scatter(ax,phase,zeros(size(phase)),32,style.slotCandidateFillColor,'filled', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.9);
hSelectedPhase = scatter(ax,phase(selectedSlot),0,90,selectedColor,'filled', ...
    'MarkerEdgeColor','k','LineWidth',1.1);
hNextPhase = scatter(ax,phase(nextSlot),0,90,nextColor,'s','filled', ...
    'MarkerEdgeColor','k','LineWidth',1.1);
hEndpoint = plot(ax,1,0,'o','MarkerSize',9,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',[0.75,0.20,0.20],'LineWidth',1.8);
plot(ax,phase([selectedSlot,nextSlot]),[0.16,0.16],'-k','LineWidth',1.5);
plot(ax,phase([selectedSlot,selectedSlot]),[0,0.16],':k');
plot(ax,phase([nextSlot,nextSlot]),[0,0.16],':k');
text(ax,mean(phase([selectedSlot,nextSlot])),0.20,'\Delta t/T=1/50', ...
    'HorizontalAlignment','center');
text(ax,0.99,-0.025,{'t=T','not stored'},'HorizontalAlignment','right', ...
    'VerticalAlignment','top');
xlabel(ax,'Normalized epoch, t/T'); yticks(ax,[]);
ylim(ax,[-0.18,0.30]); xlim(ax,[-0.02,1.02]);
style_axes(ax,style);
lgd = legend(ax,[hCandidate,hSelectedPhase,hNextPhase,hEndpoint], ...
    {'Candidate slots','Slot j','Slot j+1','Excluded endpoint'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',2,'Box','off');
style_legend(lgd,style,[16 9]);
place_metric_legend(ax,lgd,[0.12,0.19,0.80,0.56],style);
phaseFile = fullfile(outputDir,'slot_phase_grid.eps');
preview_if_requested(figPhase,inspectFigure,'endpoint-excluded phase grid');
export_manuscript_figure(figPhase,phaseFile);
close(figPhase);

orbitID = "";
if ismember('orbitID',T.Properties.VariableNames), orbitID = string(T.orbitID(orbitIndex)); end
slotSummary = table(orbitIndex,orbitID,family(orbitIndex),numSlots,period,deltaTime, ...
    min(adjacentChord_km),median(adjacentChord_km),max(adjacentChord_km), ...
    'VariableNames',{'catalogRow','orbitID','family','numSlots','period_TU', ...
    'deltaTime_TU','minimumChord_km','medianChord_km','maximumChord_km'});
summaryFile = fullfile(outputDir,'slot_definition_summary.csv');
writetable(slotSummary,summaryFile);
outputs = struct('figures',[string(geometryFile);string(phaseFile)], ...
    'geometryFigure',string(geometryFile),'phaseFigure',string(phaseFile), ...
    'summary',string(summaryFile),'slotSummary',slotSummary);
fprintf('Saved the two slot-definition manuscript figures to:\n  %s\n',outputDir);
end


function fig = manuscript_figure(widthIn,heightIn,style)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperPosition',[0 0 widthIn heightIn], ...
    'PaperSize',[widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
set(fig,'DefaultAxesFontName',style.fontName,'DefaultAxesFontSize',style.fontSize, ...
    'DefaultAxesFontWeight',style.fontWeight,'DefaultTextFontName',style.fontName, ...
    'DefaultTextFontSize',style.fontSize,'DefaultTextFontWeight',style.fontWeight);
end

function prepare_geometry_axes(ax,style)
hold(ax,'on'); box(ax,'off'); grid(ax,'off'); axis(ax,'equal');
view(ax,style.geometryAzimuth,style.geometryElevation); ax.Projection=style.geometryProjection;
xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)'); style_axes(ax,style);
end

function style_axes(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight',style.fontWeight, ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top', ...
    'Box','off','XGrid','off','YGrid','off','ZGrid','off');
ax.XLabel.FontSize=style.labelFontSize; ax.YLabel.FontSize=style.labelFontSize;
ax.ZLabel.FontSize=style.labelFontSize; ax.XLabel.FontWeight=style.fontWeight;
ax.YLabel.FontWeight=style.fontWeight; ax.ZLabel.FontWeight=style.fontWeight;
end

function style_legend(lgd,style,itemTokenSize)
lgd.FontName=style.fontName; lgd.FontSize=style.fontSize; lgd.FontWeight=style.fontWeight;
lgd.ItemTokenSize=itemTokenSize; lgd.Box='off';
end

function place_geometry_legend(ax,lgd,plotPosition,style)
lgd.Location='northoutside'; lgd.Units='normalized'; drawnow; north=lgd.Position;
lgd.Location='none'; ax.PositionConstraint='innerposition'; ax.Position=plotPosition; drawnow;
pos=lgd.Position; pos(1)=max(0.002,min(0.5-pos(3)/2,0.998-pos(3)));
pos(2)=min(max(north(2)+style.legendNorthOutsideYOffset, ...
    plotPosition(2)+plotPosition(4)+style.legendMinimumGap),0.99-pos(4));
lgd.Position=pos; lgd.AutoUpdate='off'; ax.Position=plotPosition; drawnow;
end

function place_metric_legend(ax,lgd,plotPosition,style)
place_geometry_legend(ax,lgd,plotPosition,style);
end

function simplify_geometry_ticks(ax)
for name=["X","Y","Z"]
    prop=name+"Tick"; ticks=double(ax.(prop));
    if numel(ticks)~=3 || any(~isfinite(ticks)), continue; end
    tol=100*eps(max(1,max(abs(ticks))));
    if abs(ticks(2))<=tol && abs(ticks(1)+ticks(3))<=tol
        ax.(prop)=ticks([1 3]);
    end
end
end

function lim=pad_limits(lim,fraction)
span=lim(2)-lim(1); if span<=100*eps(max(1,max(abs(lim)))), span=max(0.02,0.05*max(1,abs(mean(lim)))); end
lim=lim+[-fraction fraction]*span;
end

function h=draw_moon(ax,mu,LU)
radius=1737.1/LU; [x,y,z]=sphere(30);
h=surf(ax,radius*x+1-mu,radius*y,radius*z,'FaceColor',[0.72 0.72 0.72], ...
    'EdgeColor','none','FaceLighting','gouraud'); camlight(ax,'headlight'); material(ax,'dull');
end

function preview_if_requested(fig,inspectFigure,description)
if inspectFigure
    figure(fig); drawnow; fprintf('Previewing the %s figure for 5 seconds before export.\n',description); pause(5);
end
end
