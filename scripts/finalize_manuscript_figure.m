function finalize_manuscript_figure(fig)
%FINALIZE_MANUSCRIPT_FIGURE Apply common paper typography before export.
% The data, camera, projection, limits, colors, and annotations are preserved.
% Only axes/label/legend typography and the final fit/centering pass change.

if nargin < 1 || isempty(fig) || ~isgraphics(fig), return; end
style = reviewer2_paper_style();
axesList = findall(fig,'Type','axes');
legends = findall(fig,'Type','legend');

for k = 1:numel(axesList)
    ax = axesList(k);
    if is_3d_axes(ax)
        tickSize = style.geometryFontSize;
        labelSize = style.geometryLabelFontSize;
    else
        tickSize = style.fontSize;
        labelSize = style.labelFontSize;
    end
    set(ax,'FontName',style.fontName,'FontWeight',style.fontWeight,'FontSize',tickSize);
    labels = [ax.XLabel ax.YLabel ax.ZLabel];
    for q = 1:numel(labels)
        if isgraphics(labels(q))
            labels(q).FontName = style.fontName;
            labels(q).FontWeight = style.fontWeight;
            labels(q).FontSize = labelSize;
        end
    end
end

for k = 1:numel(legends)
    lgd = legends(k);
    if numel(axesList) == 1 && is_3d_axes(axesList(1))
        target = style.geometryLegendFontSize;
    else
        target = style.legendFontSize;
    end
    lgd.FontName = style.fontName;
    lgd.FontWeight = style.fontWeight;
    lgd.FontSize = target;
    lgd.Orientation = 'horizontal';
    lgd.NumColumns = manuscript_legend_columns(lgd,style);
    lgd.Units = 'normalized';
    drawnow;

    % Keep the two-row layout whenever possible. Reduce font size first if a
    % long legend is still wider than the standardized manuscript canvas.
    while lgd.Position(3) > style.legendMaxWidth && ...
            lgd.FontSize > style.legendMinFontSize
        lgd.FontSize = lgd.FontSize-1;
        drawnow;
    end

    % Last-resort wrapping prevents clipping while preserving the common
    % canvas. This may introduce a third row only when two rows cannot fit.
    columns = lgd.NumColumns;
    while lgd.Position(3) > style.legendMaxWidth && columns > 1
        columns = columns-1;
        lgd.NumColumns = columns;
        drawnow;
    end
end

if numel(axesList) == 1
    ax = axesList(1);
    format_manuscript_ticks(ax);
    if isempty(legends), lgd = []; else, lgd = legends(1); end
    center_manuscript_content(fig,ax,lgd);
end
drawnow;
end

function tf = is_3d_axes(ax)
tf = abs(ax.View(2)-90) > 1e-8;
if tf, return; end
objects = findall(ax,'-property','ZData');
for k = 1:numel(objects)
    z = objects(k).ZData;
    if isnumeric(z) && ~isempty(z) && any(isfinite(z(:)))
        tf = true;
        return;
    end
end
end
