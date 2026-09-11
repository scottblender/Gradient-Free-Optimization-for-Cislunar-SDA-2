function test_manuscript_figure_export()
% Data-free regression for style -> generate -> export manuscript workflow.
setup_project();
style = reviewer2_paper_style();
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>

%% Metric figure is fully finalized before export.
fig = manuscript_test_figure(style.metricFigureWidth,style.metricFigureHeight,style);
closeFigure = onCleanup(@() close(fig));
ax = axes(fig,'Units','normalized','Position',style.metricPlotPosition);
hold(ax,'on');
x = linspace(60,6000,200);
h = gobjects(3,1);
h(1)=plot(ax,x,3.6+4.2*(x/60).^(-0.35),'LineWidth',style.lineWidth);
h(2)=plot(ax,x,4.8+3.6*(x/60).^(-0.28),'LineWidth',style.lineWidth);
h(3)=plot(ax,x,5.8+3.0*(x/60).^(-0.22),'LineWidth',style.lineWidth);
xlim(ax,[60 6000]); ylim(ax,[3.4 9.2]);
ax.XTick = [60 1000 2000 3000 4000 5000 6000];
ax.YTick = [4 5 6 7 8 9];
xlabel(ax,'Function evaluations','FontWeight',style.fontWeight,'FontSize',style.labelFontSize);
ylabel(ax,'Mean best-so-far objective','FontWeight',style.fontWeight,'FontSize',style.labelFontSize);
style_test_axes(ax,style);
lgd = legend(ax,h,{'1 period','3 periods','5 periods'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',3,'Box','off');
style_test_legend(lgd,style);
drawnow;

metricPosition = ax.Position;
metricXTick = ax.XTick;
metricYTick = ax.YTick;
metricLegendPosition = lgd.Position;
metricLegendStrings = string(lgd.String);
metricFile = fullfile(folder,'metric.eps');
meta = export_manuscript_figure(fig,metricFile);

assert(isequal(ax.Position,metricPosition),'Exporter changed metric axes position.');
assert(isequal(ax.XTick,metricXTick),'Exporter changed metric x ticks.');
assert(isequal(ax.YTick,metricYTick),'Exporter changed metric y ticks.');
assert(isequal(lgd.Position,metricLegendPosition),'Exporter changed metric legend position.');
assert(isequal(string(lgd.String),metricLegendStrings),'Exporter changed legend text.');
assert(meta.minimumPrintedFontPoints>=style.minimumPrintedFontSize);
assert(isfile(metricFile) && isfile(strrep(metricFile,'.eps','.png')));
epsText = fileread(metricFile);
assert(contains(epsText,'%%HiResBoundingBox: 0 0 468.000000 417.600000'), ...
    'Metric EPS did not retain the 6.5 x 5.8 inch paper canvas.');
clear closeFigure;

%% 3-D geometry state is never mutated at export.
fig = manuscript_test_figure(style.geometryFigureWidth,style.geometryFigureHeight,style);
closeFigure = onCleanup(@() close(fig));
ax = axes(fig,'Units','normalized','Position',style.geometryPlotPosition);
hold(ax,'on'); axis(ax,'equal');
t = linspace(0,2*pi,200);
h1=plot3(ax,0.95+0.12*cos(t),0.04*sin(t),0.18*sin(t),'-','LineWidth',1.5);
h2=plot3(ax,0.96+0.10*cos(t),0.03*sin(t),0.15*sin(t),'-','LineWidth',1.5);
view(ax,-37.5,30); ax.Projection='perspective'; axis(ax,'vis3d');
xlabel(ax,'x (LU)','FontWeight',style.fontWeight,'FontSize',style.labelFontSize);
ylabel(ax,'y (LU)','FontWeight',style.fontWeight,'FontSize',style.labelFontSize);
zlabel(ax,'z (LU)','FontWeight',style.fontWeight,'FontSize',style.labelFontSize);
style_test_axes(ax,style);
ax.YTick=[-0.05 0.05];
lgd=legend(ax,[h1,h2],{'L1','L2'},'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',2,'Box','off');
style_test_legend(lgd,style);
drawnow;

geometryPosition=ax.Position;
geometryTicks=ax.YTick;
geometryView=view(ax);
geometryCamera=ax.CameraPosition;
geometryFile=fullfile(folder,'geometry.eps');
export_manuscript_figure(fig,geometryFile);
assert(isequal(ax.Position,geometryPosition),'Exporter changed 3-D axes position.');
assert(isequal(ax.YTick,geometryTicks),'Exporter changed 3-D trajectory ticks.');
assert(max(abs(view(ax)-geometryView))<1e-12,'Exporter changed 3-D view.');
assert(max(abs(ax.CameraPosition-geometryCamera))<1e-12,'Exporter changed 3-D camera.');
assert(isfile(geometryFile) && dir(geometryFile).bytes>0);
clear closeFigure;

%% Visibility schematic uses a genuinely taller EPS canvas and tighter limits.
assert(style.visibilityFigureHeight>style.geometryFigureHeight);
assert(diff(style.visibilityXLim)<6.45 && diff(style.visibilityYLim)<5.60, ...
    'Visibility limits are not tighter than the legacy schematic limits.');
visibility = plot_visibility_keepout_manuscript(false,folder);
assert(isfile(visibility.figure));
epsText = fileread(visibility.figure);
assert(contains(epsText,'%%HiResBoundingBox: 0 0 468.000000 460.800000'), ...
    'Visibility EPS did not retain the 6.5 x 6.4 inch paper canvas.');

fprintf(['Manuscript export checks passed: generation-time styling is preserved, ' ...
    '3-D state is not mutated, metric canvas has label/legend room, and the ' ...
    'visibility schematic uses the larger final EPS geometry.\n']);
end


function fig = manuscript_test_figure(widthIn,heightIn,style)
fig = figure('Visible','off','Color','w','Units','inches', ...
    'Position',[1 1 widthIn heightIn],'PaperUnits','inches', ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperSize',[widthIn heightIn], ...
    'PaperPositionMode','manual','Renderer','painters','InvertHardcopy','off');
set(fig,'DefaultAxesFontName',style.fontName,'DefaultAxesFontSize',style.fontSize, ...
    'DefaultAxesFontWeight',style.fontWeight,'DefaultTextFontName',style.fontName, ...
    'DefaultTextFontSize',style.fontSize,'DefaultTextFontWeight',style.fontWeight);
end

function style_test_axes(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
    'FontWeight',style.fontWeight,'LineWidth',style.axisLineWidth, ...
    'TickDir','out','Box','off','XGrid','off','YGrid','off','ZGrid','off');
end

function style_test_legend(lgd,style)
lgd.FontName=style.fontName;
lgd.FontSize=style.fontSize;
lgd.FontWeight=style.fontWeight;
lgd.Box='off';
end
