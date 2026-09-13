function files = export_shared_result_legend(outputDirectory,stem,labels,colors,style,varargin)
%EXPORT_SHARED_RESULT_LEGEND Export a legend-only EPS/PNG for subfigure groups.
%
% The data panels remain legend-free and use the larger no-legend axes box.
% LaTeX places this compact legend strip above the corresponding subfigures,
% matching the trajectory-grid and Monte Carlo manuscript layouts.

p = inputParser;
addParameter(p,'Kind','line',@(x) ischar(x) || isstring(x));
addParameter(p,'LineStyles',strings(0,1),@(x) ischar(x) || isstring(x) || iscellstr(x));
addParameter(p,'Markers',strings(0,1),@(x) ischar(x) || isstring(x) || iscellstr(x));
addParameter(p,'NumColumns',[],@(x) isempty(x) || (isscalar(x) && x >= 1));
parse(p,varargin{:});

labels = string(labels(:));
colors = double(colors);
assert(size(colors,1) == numel(labels) && size(colors,2) == 3, ...
    'colors must contain one RGB row per legend label.');

lineStyles = string(p.Results.LineStyles(:));
if isempty(lineStyles)
    lineStyles = repmat("-",numel(labels),1);
end
assert(numel(lineStyles) == numel(labels), ...
    'LineStyles must contain one entry per legend label.');

markers = string(p.Results.Markers(:));
if isempty(markers)
    markers = repmat("none",numel(labels),1);
end
assert(numel(markers) == numel(labels), ...
    'Markers must contain one entry per legend label.');

nCols = p.Results.NumColumns;
if isempty(nCols)
    nCols = min(3,ceil(numel(labels)/2));
end

fig = figure('Visible','off','Color','w','Units','inches', ...
    'Position',[1 1 style.sharedResultLegendWidth style.sharedResultLegendHeight], ...
    'PaperUnits','inches', ...
    'PaperSize',[style.sharedResultLegendWidth style.sharedResultLegendHeight], ...
    'PaperPosition',[0 0 style.sharedResultLegendWidth style.sharedResultLegendHeight], ...
    'PaperPositionMode','manual','Renderer','painters','InvertHardcopy','off');
cleanup = onCleanup(@() close(fig));
ax = axes(fig,'Units','normalized','Position',[0.01 0.01 0.98 0.98], ...
    'Visible','off');
hold(ax,'on');

handles = gobjects(numel(labels),1);
kind = lower(string(p.Results.Kind));
for k = 1:numel(labels)
    if kind == "patch"
        handles(k) = plot(ax,nan,nan,'s','LineStyle','none', ...
            'MarkerSize',10,'MarkerFaceColor',colors(k,:), ...
            'MarkerEdgeColor','none','DisplayName',labels(k));
    else
        handles(k) = plot(ax,nan,nan,'LineStyle',lineStyles(k), ...
            'Marker',markers(k),'Color',colors(k,:), ...
            'LineWidth',style.lineWidth,'MarkerSize',9, ...
            'MarkerFaceColor',colors(k,:),'MarkerEdgeColor',colors(k,:), ...
            'DisplayName',labels(k));
    end
end

lgd = legend(ax,handles,cellstr(labels),'Location','none', ...
    'Orientation','horizontal','NumColumns',nCols,'Box','off');
lgd.FontName = style.fontName;
lgd.FontSize = style.sharedLegendFontSize;
lgd.FontWeight = style.fontWeight;
lgd.ItemTokenSize = style.geometryLegendItemTokenSize;
lgd.Units = 'normalized';
drawnow;
pos = lgd.Position;
pos(1) = 0.5-pos(3)/2;
pos(2) = 0.5-pos(4)/2;
lgd.Position = pos;
lgd.AutoUpdate = 'off';
axis(ax,'off');
drawnow;

base = fullfile(char(outputDirectory),char(stem));
files = [string(base)+".eps";string(base)+".png"];
print(fig,char(files(1)),'-depsc2','-painters','-r600','-loose');
exportgraphics(fig,char(files(2)),'Resolution',style.exportDpi);
clear cleanup;
end
