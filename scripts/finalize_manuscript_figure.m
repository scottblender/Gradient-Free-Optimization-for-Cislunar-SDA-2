function finalize_manuscript_figure(fig)
%FINALIZE_MANUSCRIPT_FIGURE Apply common paper typography before export.
% The data, camera, projection, limits, colors, and annotations are preserved.
% Only axes/label/legend typography and the final fit/centering pass change.

if nargin < 1 || isempty(fig) || ~isgraphics(fig), return; end
style = reviewer2_paper_style();
axesList = findall(fig,'Type','axes');
legends = findall(fig,'Type','legend');

areas = zeros(numel(axesList),1);
for k = 1:numel(axesList)
    oldUnits = axesList(k).Units;
    axesList(k).Units = 'normalized';
    p = axesList(k).Position;
    areas(k) = p(3)*p(4);
    axesList(k).Units = oldUnits;
end
if isempty(areas)
    mainAxesIndex = [];
else
    [~,mainAxesIndex] = max(areas);
end

for k = 1:numel(axesList)
    ax = axesList(k);
    isInset = numel(axesList) > 1 && k ~= mainAxesIndex && ...
        areas(k) < 0.75*areas(mainAxesIndex);

    if is_3d_axes(ax)
        tickSize = style.geometryFontSize;
        labelSize = style.geometryLabelFontSize;
    else
        tickSize = style.fontSize;
        labelSize = style.labelFontSize;
    end

    if isInset
        tickSize = min(12,tickSize);
        labelSize = tickSize;
        ax.XTickLabel = [];
        ax.YTickLabel = [];
        ax.ZTickLabel = [];

        if has_zoom_annotation(ax)
            oldUnits = ax.Units;
            ax.Units = 'normalized';
            if is_3d_axes(ax)
                ax.Position = [0.18 0.36 0.38 0.36];
            else
                ax.Position = [0.63 0.40 0.28 0.20];
            end
            ax.Units = oldUnits;
        end

        insetText = findall(ax,'Type','text');
        for t = 1:numel(insetText)
            try
                insetText(t).FontSize = min(insetText(t).FontSize,12);
                insetText(t).FontWeight = style.fontWeight;
            catch
            end
        end
    elseif numel(axesList) > 1 && k == mainAxesIndex && ...
            is_3d_axes(ax) && numel(ax.XTick) > 2
        ax.XTick = ax.XTick([1 end]);
    end

    set(ax,'FontName',style.fontName,'FontWeight',style.fontWeight,'FontSize',tickSize);
    labels = [ax.XLabel ax.YLabel ax.ZLabel];
    for q = 1:numel(labels)
        if isgraphics(labels(q))
            labels(q).FontName = style.fontName;
            labels(q).FontWeight = style.fontWeight;
            labels(q).FontSize = labelSize;
            if isInset, labels(q).String = ''; end
        end
    end

    ax.XTickLabel = replace_abc_text(ax.XTickLabel);
    ax.YTickLabel = replace_abc_text(ax.YTickLabel);
    ax.ZTickLabel = replace_abc_text(ax.ZTickLabel);
end

textObjects = findall(fig,'-property','String');
for k = 1:numel(textObjects)
    try
        textObjects(k).String = replace_abc_text(textObjects(k).String);
    catch
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

    while lgd.Position(3) > style.legendMaxWidth && ...
            lgd.FontSize > style.legendMinFontSize
        lgd.FontSize = lgd.FontSize-1;
        drawnow;
    end

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

function tf = has_zoom_annotation(ax)
tf = false;
textObjects = findall(ax,'Type','text');
for k = 1:numel(textObjects)
    try
        value = string(textObjects(k).String);
        if any(strcmpi(strtrim(value),"Zoom"))
            tf = true;
            return;
        end
    catch
    end
end
end

function value = replace_abc_text(value)
%REPLACE_ABC_TEXT Convert the internal solver key ABC to manuscript ABCO.
if isempty(value), return; end
try
    if ischar(value)
        if isrow(value)
            value = char(regexprep(string(value), ...
                '(?<![A-Za-z])ABC(?![A-Za-z])','ABCO'));
        else
            s = string(cellstr(value));
            s = regexprep(s,'(?<![A-Za-z])ABC(?![A-Za-z])','ABCO');
            value = char(s);
        end
    elseif iscell(value)
        s = string(value);
        s = regexprep(s,'(?<![A-Za-z])ABC(?![A-Za-z])','ABCO');
        value = cellstr(s);
    elseif isstring(value)
        value = regexprep(value,'(?<![A-Za-z])ABC(?![A-Za-z])','ABCO');
    end
catch
end
end