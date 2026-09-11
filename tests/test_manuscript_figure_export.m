function test_manuscript_figure_export()
% Data-free regression for the centralized manuscript export formatter.
paths = setup_project(); %#ok<NASGU>
style = reviewer2_paper_style();
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
boxes = strings(2,1); imageSizes = zeros(2,2);

% -------------------------------------------------------------------------
% Metric figure: dense numerical ticks, readable font, adjusted northoutside.
% -------------------------------------------------------------------------
fig = figure('Visible','off','Units','inches', ...
    'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
    'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight]);
closeFigure = onCleanup(@() close(fig));
ax = axes(fig);
x = linspace(60,6000,200);
h = gobjects(3,1);
h(1)=plot(ax,x,3.6+4.2*(x/60).^(-0.35),'LineWidth',1.5); hold(ax,'on');
h(2)=plot(ax,x,4.8+3.6*(x/60).^(-0.28),'LineWidth',1.5);
h(3)=plot(ax,x,5.8+3.0*(x/60).^(-0.22),'LineWidth',1.5);
xlim(ax,[60 6000]); ylim(ax,[3.4 9.2]);
xlabel(ax,'Function evaluations'); ylabel(ax,'Mean best-so-far objective');
lgd = legend(ax,h,{'1 period','3 periods','5 periods'}, ...
    'Location','northoutside','Orientation','horizontal');
metricFile = fullfile(folder,'metric.eps');
meta = export_manuscript_figure(fig,metricFile);
assert(numel(ax.XTick)>=5 && numel(ax.XTick)<=style.max2DXTicks, ...
    'Metric x axis does not have enough readable ticks.');
assert(numel(ax.YTick)>=4 && numel(ax.YTick)<=style.max2DYTicks, ...
    'Metric y axis does not have enough readable ticks.');
assert(isappdata(ax,'ManuscriptNorthOutsideReference'), ...
    'Master formatter did not establish a northoutside legend reference.');
northPosition = getappdata(ax,'ManuscriptNorthOutsideReference');
finalPosition = lgd.Position;
assert(finalPosition(2)<=northPosition(2)+1e-6, ...
    'Legend was not moved downward from northoutside.');
assert(finalPosition(2)>=ax.Position(2)+ax.Position(4)+style.legendMinimumGap-0.003, ...
    'Legend is too close to or overlapping the metric axes.');
assert(strcmpi(lgd.Location,'none'), ...
    'Legend should be frozen only after adjusted northoutside placement.');
assert(meta.minimumPrintedFontPoints>=style.minimumPrintedFontSize);
check_fonts(fig,style);
check_canvas(fig);
epsText = fileread(metricFile);
boxes(1) = string(regexp(epsText,'(?m)^%%BoundingBox:[^\r\n]*','match','once'));
assert(contains(epsText,'%%HiResBoundingBox: 0 0 468.000000 374.400000'));
png = imfinfo(strrep(metricFile,'.eps','.png')); imageSizes(1,:)=[png.Width png.Height];
clear closeFigure;

% -------------------------------------------------------------------------
% Geometry figure: remove only a symmetric center-zero tick that can overlap
% after 3-D projection; nonsymmetric trajectory tick sets remain untouched.
% -------------------------------------------------------------------------
fig = figure('Visible','off','Units','inches', ...
    'Position',[1 1 style.geometryFigureWidth style.geometryFigureHeight], ...
    'PaperUnits','inches','PaperSize',[style.geometryFigureWidth style.geometryFigureHeight]);
closeFigure = onCleanup(@() close(fig));
ax = axes(fig,'Units','normalized','Position',style.geometryPlotPosition);
t = linspace(0,2*pi,200);
h1=plot3(ax,0.95+0.12*cos(t),0.04*sin(t),0.18*sin(t),'-','LineWidth',1.5); hold(ax,'on');
h2=plot3(ax,0.96+0.10*cos(t),0.03*sin(t),0.15*sin(t),'-','LineWidth',1.5);
xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
view(ax,-37.5,30); axis(ax,'equal');
ax.XTick=[0.85 0.95 1.05]; originalXTicks=ax.XTick;
ax.YTick=[-0.05 0 0.05];
lgd=legend(ax,[h1,h2],{'L1','L2'},'Location','northoutside','Orientation','horizontal');
geometryFile = fullfile(folder,'geometry.eps');
export_manuscript_figure(fig,geometryFile);
assert(isequal(ax.YTick,[-0.05 0.05]), ...
    'Symmetric trajectory ticks should drop the overlapping center zero.');
assert(isequal(ax.XTick,originalXTicks), ...
    'Nonsymmetric trajectory ticks should remain unchanged.');
assert(isappdata(ax,'ManuscriptNorthOutsideReference'));
northPosition = getappdata(ax,'ManuscriptNorthOutsideReference');
assert(lgd.Position(2)<=northPosition(2)+1e-6, ...
    'Geometry legend was not moved downward from northoutside.');
check_fonts(fig,style);
check_canvas(fig);
epsText = fileread(geometryFile);
boxes(2) = string(regexp(epsText,'(?m)^%%BoundingBox:[^\r\n]*','match','once'));
png = imfinfo(strrep(geometryFile,'.eps','.png')); imageSizes(2,:)=[png.Width png.Height];
clear closeFigure;

% -------------------------------------------------------------------------
% Low-thrust geometry legend: the real comparison panel has eight entries.
% At full manuscript font size it must fit the fixed EPS canvas without
% changing the font size or creating more than two legend rows.
% -------------------------------------------------------------------------
fig = figure('Visible','off','Units','inches', ...
    'Position',[1 1 style.geometryFigureWidth style.geometryFigureHeight], ...
    'PaperUnits','inches','PaperSize',[style.geometryFigureWidth style.geometryFigureHeight]);
closeFigure = onCleanup(@() close(fig));
ax = axes(fig,'Units','normalized','Position',style.geometryPlotPosition); hold(ax,'on');
t = linspace(0,2*pi,80);
h = gobjects(8,1);
for k = 1:8
    h(k)=plot3(ax,0.95+0.04*k/8*cos(t),0.02*sin(t),0.05*sin(t+k/10), ...
        'LineWidth',1.2);
end
xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
view(ax,-37.5,35); axis(ax,'equal');
lgd = legend(ax,h,{'Endpoint orbits','Target trajectory','Observer orbits', ...
    'Start','End','Moon','L1','L2'},'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',4);
longLegendFile = fullfile(folder,'long_geometry_legend.eps');
export_manuscript_figure(fig,longLegendFile);
labels = string(lgd.String);
assert(any(labels=="Endpoints") && any(labels=="Target") && ...
    any(labels=="Obs. orbits"), ...
    'Standard long trajectory legend labels were not compacted.');
assert(ceil(numel(labels)/lgd.NumColumns)<=style.legendMaxRows, ...
    'Long trajectory legend requires too many rows.');
check_fonts(fig,style);
check_canvas(fig);
clear closeFigure;

% -------------------------------------------------------------------------
% Slot demonstration: candidate markers become filled but excluded endpoint
% remains hollow so the slot convention is still visually unambiguous.
% -------------------------------------------------------------------------
fig = figure('Visible','off','Units','inches', ...
    'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
    'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight]);
closeFigure = onCleanup(@() close(fig));
ax = axes(fig); hold(ax,'on');
hCandidate = plot(ax,1:5,zeros(1,5),'o','MarkerSize',5, ...
    'MarkerFaceColor','w','MarkerEdgeColor',[0.25 0.25 0.25]);
hSelected = plot(ax,3,0,'o','MarkerSize',9,'MarkerFaceColor',[0.85 0.25 0.20], ...
    'MarkerEdgeColor','k');
hNext = plot(ax,4,0,'s','MarkerSize',9,'MarkerFaceColor',[0.20 0.50 0.80], ...
    'MarkerEdgeColor','k');
hEndpoint = plot(ax,6,0,'o','MarkerSize',9,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',[0.75 0.20 0.20]);
legend(ax,[hCandidate,hSelected,hNext,hEndpoint], ...
    {'Candidate slots','Slot j','Slot j+1','Excluded endpoint'}, ...
    'Location','northoutside','Orientation','horizontal');
xlim(ax,[0 6.5]); ylim(ax,[-1 1]);
slotFile = fullfile(folder,'slot_demo.eps');
export_manuscript_figure(fig,slotFile);
assert(isnumeric(hCandidate.MarkerFaceColor) && ...
    max(abs(double(hCandidate.MarkerFaceColor)-style.slotCandidateFillColor))<1e-12, ...
    'Candidate slot markers were not filled for manuscript visibility.');
assert(ischar(hEndpoint.MarkerFaceColor) || isstring(hEndpoint.MarkerFaceColor), ...
    'Excluded endpoint should remain a hollow white marker.');
check_canvas(fig);
clear closeFigure;

assert(boxes(1)==boxes(2),'Paired EPS canvases differ.');
assert(isequal(imageSizes(1,:),imageSizes(2,:)),'Paired PNG canvases differ.');
assert(style.geometryFigureWidth==style.measurementFigureWidth && ...
    style.geometryFigureHeight==style.measurementFigureHeight);
assert(style.visibilityFigureWidth==style.metricFigureWidth && ...
    style.visibilityFigureHeight>style.metricFigureHeight, ...
    'Visibility/occlusion schematic should use a taller canvas at common width.');
assert(style.fontSize*style.manuscriptPanelWidth/style.metricFigureWidth >= ...
    style.minimumPrintedFontSize,'Configured manuscript font is too small after placement.');
assert(style.legendNorthOutsideYOffset<0 && style.legendMinimumGap>0);
fprintf(['Centralized manuscript formatter checks passed: readable fonts, adjusted ' ...
    'northoutside legends, metric ticks, compact symmetric trajectory ticks, ' ...
    'long trajectory legend fit, filled slot candidates, taller keepout schematic, ' ...
    'and EPS fit.\n']);
end


function check_fonts(fig,style)
objects=findall(fig,'-property','FontSize');
assert(all(arrayfun(@(obj) obj.FontSize>=style.fontSize,objects)), ...
    'A manuscript graphics object is below the configured font size.');
weights=findall(fig,'-property','FontWeight');
assert(all(arrayfun(@(obj) strcmpi(obj.FontWeight,style.fontWeight),weights)), ...
    'All manuscript text should use the shared font weight.');
end


function check_canvas(fig)
axesObjects=findall(fig,'Type','axes');
for k=1:numel(axesObjects)
    ax=axesObjects(k); if strcmpi(ax.Visible,'off'), continue; end
    p=ax.Position;
    assert(p(1)>=-0.005 && p(2)>=-0.005 && p(1)+p(3)<=1.005 && p(2)+p(4)<=1.005);
    if ~isempty(ax.Legend)
        lp=ax.Legend.Position;
        assert(lp(1)>=-0.005 && lp(2)>=-0.005 && lp(1)+lp(3)<=1.005 && lp(2)+lp(4)<=1.005);
    end
end
end
