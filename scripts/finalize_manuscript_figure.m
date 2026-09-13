function finalize_manuscript_figure(fig)
%FINALIZE_MANUSCRIPT_FIGURE Apply common paper typography before export.
% The data, camera, projection, limits, colors, annotations, and plot-specific
% axes positions are preserved. Only typography and final fit/centering change.

if nargin < 1 || isempty(fig) || ~isgraphics(fig), return; end
style = reviewer2_paper_style();
axesList = findall(fig,'Type','axes');
legends = findall(fig,'Type','legend');

% The nominal Lunar Gateway perspective view needs a little more physical
% paper around the otherwise standard manuscript panel. Expand only its outer
% canvas by 0.25 in per side while preserving the physical size of the axes
% and legend. The flag makes this safe because the export path finalizes twice.
if is_lunar_gateway_case(legends)
    expand_lunar_gateway_canvas(fig,axesList,legends);
end

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

        % Preserve plot-specific 3-D inset placement. The plotting routine
        % intentionally chooses the 3-D zoom position, so the common export
        % finalizer must not overwrite it. Retain the legacy placement only
        % for 2-D zoom insets that rely on the common formatter.
        if has_zoom_annotation(ax) && ~is_3d_axes(ax)
            oldUnits = ax.Units;
            ax.Units = 'normalized';
            ax.Position = [0.63 0.40 0.28 0.20];
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

function expand_lunar_gateway_canvas(fig,axesList,legends)
%EXPAND_LUNAR_GATEWAY_CANVAS Add real paper around the nominal LG panel.
% The normal canvas is 6.5 x 5.2 in. Adding 0.25 in on all four sides gives
% 7.0 x 5.7 in without scaling the axes, labels, or legend themselves.

appDataKey = 'LunarGatewayCanvasExpanded';
if isappdata(fig,appDataKey) && getappdata(fig,appDataKey)
    return;
end

padInches = 0.25;
figUnits = fig.Units;
fig.Units = 'inches';
figurePosition = fig.Position;
newWidth = figurePosition(3)+2*padInches;
newHeight = figurePosition(4)+2*padInches;

axesUnits = cell(numel(axesList),1);
for k = 1:numel(axesList)
    axesUnits{k} = axesList(k).Units;
    axesList(k).Units = 'inches';
    p = axesList(k).Position;
    p(1:2) = p(1:2)+padInches;
    axesList(k).Position = p;
end

legendUnits = cell(numel(legends),1);
for k = 1:numel(legends)
    legendUnits{k} = legends(k).Units;
    legends(k).Units = 'inches';
    p = legends(k).Position;
    p(1:2) = p(1:2)+padInches;
    legends(k).Position = p;
end

figurePosition(3:4) = [newWidth,newHeight];
fig.Position = figurePosition;
fig.PaperUnits = 'inches';
fig.PaperSize = [newWidth,newHeight];
fig.PaperPosition = [0,0,newWidth,newHeight];
fig.PaperPositionMode = 'manual';

for k = 1:numel(axesList)
    axesList(k).Units = axesUnits{k};
end
for k = 1:numel(legends)
    legends(k).Units = legendUnits{k};
end
fig.Units = figUnits;
setappdata(fig,appDataKey,true);
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

function tf = is_lunar_gateway_case(legends)
tf = false;
if numel(legends) ~= 1 || ~isgraphics(legends(1)), return; end
try
    labels = strtrim(string(legends(1).String(:)));
    tf = isequal(labels,["Nominal Gateway";"Moon";"L1";"L2"]);
catch
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
