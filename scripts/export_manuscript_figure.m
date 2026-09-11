function metadata = export_manuscript_figure(fig,fileName)
%EXPORT_MANUSCRIPT_FIGURE Apply manuscript style once, then export EPS/PNG.
% All export-time typography, legend placement, metric tick density, and
% canvas-fit checks are centralized here. No second layout formatter runs.

style = reviewer2_paper_style();
[folder,stem,~] = fileparts(char(fileName));
if isempty(folder), folder = pwd; end
if ~isfolder(folder), mkdir(folder); end
base = fullfile(folder,stem);

set(fig,'PaperUnits','inches','PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off','Color','w');
paper = get(fig,'PaperSize');
set(fig,'PaperPosition',[0 0 paper]);

format_manuscript_figure(fig,style);
assert_canvas_fit(fig,stem);

% Fail visibly on unsupported transparency rather than producing a subtly
% different EPS. Current manuscript renderers use opaque vector objects.
for property = ["FaceAlpha","EdgeAlpha"]
    objects = findall(fig,'-property',char(property));
    for k = 1:numel(objects)
        value = get(objects(k),char(property));
        assert(isnumeric(value) && all(value(:)==1), ...
            'Manuscript:Transparency','EPS requires opaque %s in %s.',property,stem);
    end
end

drawnow;
print(fig,[base '.eps'],'-depsc2','-painters','-loose');

% Normalize the EPS bounding boxes to the physical paper rectangle so every
% paired panel scales identically in LaTeX and no export grows unexpectedly.
epsText = fileread([base '.eps']);
assert(startsWith(epsText,'%!PS-Adobe'),'Invalid EPS output: %s',base);
widthPt = 72*paper(1); heightPt = 72*paper(2);
box = sprintf('%%%%BoundingBox: 0 0 %d %d',ceil(widthPt),ceil(heightPt));
hires = sprintf('%%%%HiResBoundingBox: 0 0 %.6f %.6f',widthPt,heightPt);
epsText = regexprep(epsText,'(?m)^%%BoundingBox:[^\r\n]*',box);
if contains(epsText,'%%HiResBoundingBox:')
    epsText = regexprep(epsText,'(?m)^%%HiResBoundingBox:[^\r\n]*',hires);
else
    firstNewline = find(epsText==newline,1);
    epsText = [epsText(1:firstNewline) hires newline epsText(firstNewline+1:end)];
end
fid = fopen([base '.eps'],'w');
assert(fid~=-1,'Cannot write EPS: %s',base);
cleanup = onCleanup(@() fclose(fid));
fwrite(fid,epsText,'char');
clear cleanup;

% print uses PaperPosition for PNG too; exportgraphics would tightly crop.
print(fig,[base '.png'],'-dpng',sprintf('-r%d',style.exportDpi));
metadata = struct('stem',stem,'widthInches',paper(1), ...
    'heightInches',paper(2),'minimumFontPoints',style.fontSize, ...
    'placementWidthInches',style.manuscriptPanelWidth, ...
    'minimumPrintedFontPoints',style.fontSize*style.manuscriptPanelWidth/paper(1));
end


function format_manuscript_figure(fig,style)
%FORMAT_MANUSCRIPT_FIGURE Single authoritative pre-export formatter.

% Improve neutral candidate-slot visibility only for the slot-definition
% demonstration. Selected/adjacent slots already use distinct filled colors;
% the excluded endpoint remains hollow by design.
fill_slot_demo_candidate_markers(fig,style);

% Typography is finalized first so MATLAB measures legends at the actual
% manuscript font size. No later helper is allowed to resize or reposition.
fontObjects = findall(fig,'-property','FontSize');
for k = 1:numel(fontObjects)
    obj = fontObjects(k);
    if isprop(obj,'FontUnits'), obj.FontUnits = 'points'; end
    obj.FontSize = max(obj.FontSize,style.fontSize);
    if isprop(obj,'FontName'), obj.FontName = style.fontName; end
    if isprop(obj,'FontWeight'), obj.FontWeight = style.fontWeight; end
end

axesObjects = findall(fig,'Type','axes');
for k = 1:numel(axesObjects)
    ax = axesObjects(k);
    if strcmpi(ax.Visible,'off'), continue; end
    ax.Units = 'normalized';
    ax.FontName = style.fontName;
    ax.FontSize = max(ax.FontSize,style.fontSize);
    ax.FontWeight = style.fontWeight;
    ax.LineWidth = style.axisLineWidth;
    ax.Box = 'off';
    ax.XGrid = 'off'; ax.YGrid = 'off'; ax.ZGrid = 'off';

    if isappdata(ax,'ManuscriptAxesPosition')
        basePosition = getappdata(ax,'ManuscriptAxesPosition');
    else
        basePosition = ax.Position;
    end

    % Add useful numerical resolution only to non-trajectory metric plots.
    % CR3BP geometry/trajectory plots are identified by 3-D view or explicit
    % '(LU)' axis units and retain their plotter-selected ticks exactly.
    if ~is_geometry_axis(ax)
        densify_metric_ticks(ax,'X',style.max2DXTicks);
        densify_metric_ticks(ax,'Y',style.max2DYTicks);
    end

    lgd = ax.Legend;
    if isempty(lgd) || ~isvalid(lgd)
        ax.PositionConstraint = 'innerposition';
        ax.Position = basePosition;
        continue;
    end

    % Start from MATLAB's correct northoutside geometry, then nudge the
    % legend slightly downward to reduce unused EPS whitespace. The final
    % position is frozen only after MATLAB has established northoutside.
    lgd.Units = 'normalized';
    lgd.Box = 'off';
    lgd.FontName = style.fontName;
    lgd.FontSize = max(lgd.FontSize,style.fontSize);
    lgd.FontWeight = style.fontWeight;
    lgd.Orientation = 'horizontal';
    lgd.Location = 'northoutside';

    count = numel(lgd.String);
    if count <= style.legendMaxColumns
        columns = max(1,count);
    else
        columns = ceil(count/style.legendMaxRows);
    end
    lgd.NumColumns = columns;
    drawnow;

    % A one-row legend that is too wide becomes a balanced two-row legend.
    pos = lgd.Position;
    if pos(3) > style.legendWidthLimit && count > 2 && columns == count
        columns = ceil(count/2);
        lgd.NumColumns = columns;
        drawnow;
        pos = lgd.Position;
    end

    % Preserve the final font size. If necessary, compact only the legend
    % sample swatches so the EPS remains inside the fixed canvas.
    if pos(3) > style.legendWidthLimit && isprop(lgd,'ItemTokenSize')
        token = lgd.ItemTokenSize;
        while pos(3) > style.legendWidthLimit && token(1) > 8
            token(1) = max(8,token(1)-2);
            lgd.ItemTokenSize = token;
            drawnow;
            pos = lgd.Position;
        end
    end

    rows = ceil(count/max(1,lgd.NumColumns));
    assert(rows <= style.legendMaxRows,'Manuscript:LegendRows', ...
        'Legend requires more than %d rows in %s.',style.legendMaxRows,class(lgd));

    northPosition = lgd.Position;
    setappdata(ax,'ManuscriptNorthOutsideReference',northPosition);
    lgd.Location = 'none';

    % Restore the plotter's intended axes rectangle after northoutside has
    % performed its automatic sizing, then place the legend relative to it.
    ax.PositionConstraint = 'innerposition';
    ax.Position = basePosition;
    drawnow;
    pos = lgd.Position;
    pos(1) = max(0.002,(1-pos(3))/2);
    minimumBottom = basePosition(2)+basePosition(4)+style.legendMinimumGap;
    desiredBottom = northPosition(2)+style.legendNorthOutsideYOffset;
    maximumBottom = 0.99-pos(4);
    pos(2) = min(max(desiredBottom,minimumBottom),maximumBottom);
    lgd.Position = pos;
    ax.Position = basePosition;
    setappdata(ax,'ManuscriptFinalLegendPosition',pos);
end

drawnow;
end


function fill_slot_demo_candidate_markers(fig,style)
%FILL_SLOT_DEMO_CANDIDATE_MARKERS Fill neutral slot-grid markers for print.
legends = findall(fig,'Type','legend');
isSlotDemo = false;
for k = 1:numel(legends)
    labels = string(legends(k).String);
    if any(labels == "Candidate slots") && any(labels == "Slot j")
        isSlotDemo = true;
        break;
    end
end
if ~isSlotDemo, return; end

% 3-D slot-geometry panel: the candidate slots are the small hollow circles;
% Slot j and Slot j+1 are larger and already filled with distinct colors.
lines = findall(fig,'Type','line');
for k = 1:numel(lines)
    h = lines(k);
    if ~isprop(h,'Marker') || strcmpi(string(h.Marker),"none") || ...
            ~isprop(h,'MarkerFaceColor') || ~isprop(h,'MarkerSize')
        continue;
    end
    if h.MarkerSize <= 6 && is_white_color(h.MarkerFaceColor)
        h.MarkerFaceColor = style.slotCandidateFillColor;
    end
end

% Phase-grid panel: scatter(...,'w','filled') may store white in CData with
% MarkerFaceColor='flat', so handle that representation explicitly.
scatters = findall(fig,'Type','scatter');
for k = 1:numel(scatters)
    h = scatters(k);
    if ~isprop(h,'SizeData') || isempty(h.SizeData) || max(double(h.SizeData(:))) > 40
        continue;
    end
    faceIsWhite = isprop(h,'MarkerFaceColor') && is_white_color(h.MarkerFaceColor);
    cdataIsWhite = isprop(h,'CData') && is_white_matrix(h.CData);
    if faceIsWhite || cdataIsWhite
        if isprop(h,'CData'), h.CData = style.slotCandidateFillColor; end
        if isprop(h,'MarkerFaceColor'), h.MarkerFaceColor = style.slotCandidateFillColor; end
    end
end
end


function tf = is_white_color(value)
if ischar(value) || isstring(value)
    tf = any(strcmpi(string(value),["w","white"]));
elseif isnumeric(value) && numel(value)==3
    tf = all(abs(double(value(:).')-[1 1 1]) < 1e-12);
else
    tf = false;
end
end


function tf = is_white_matrix(value)
tf = isnumeric(value) && ~isempty(value) && all(abs(double(value(:))-1) < 1e-12);
end


function tf = is_geometry_axis(ax)
% Geometry/trajectory axes keep their original tick choices.
viewAngles = view(ax);
isPerspective3D = abs(viewAngles(1)) > 1e-9 || abs(viewAngles(2)-90) > 1e-9;
labels = [label_text(ax.XLabel),label_text(ax.YLabel),label_text(ax.ZLabel)];
hasLU = any(contains(lower(labels),'(lu)'));
tf = isPerspective3D || hasLU;
end


function value = label_text(labelHandle)
value = string(labelHandle.String);
if isempty(value), value = ""; else, value = strjoin(value(:).'," "); end
end


function densify_metric_ticks(ax,axisName,maxTicks)
% Add readable nice-number ticks only to automatic linear numeric axes.
if axisName == "X"
    scale = ax.XScale; tickMode = ax.XTickMode; limits = ax.XLim; ticks = ax.XTick;
else
    scale = ax.YScale; tickMode = ax.YTickMode; limits = ax.YLim; ticks = ax.YTick;
end
if ~strcmpi(scale,'linear') || ~strcmpi(tickMode,'auto'), return; end
niceTicks = nice_linear_ticks(limits,maxTicks);
if isempty(niceTicks), return; end
if numel(niceTicks) > numel(ticks) || numel(ticks) > maxTicks
    if axisName == "X", ax.XTick = niceTicks; else, ax.YTick = niceTicks; end
end
end


function ticks = nice_linear_ticks(limits,maxTicks)
limits = double(limits(:).'); ticks = [];
if numel(limits)~=2 || any(~isfinite(limits)) || limits(2)<=limits(1), return; end
span = limits(2)-limits(1);
roughStep = span/max(2,maxTicks-1);
if ~isfinite(roughStep) || roughStep<=0, return; end
power = 10^floor(log10(roughStep));
steps = power*[1 2 2.5 5 10];
tolerance = 1e-10*max(1,max(abs(limits)));
for step = steps
    first = ceil((limits(1)-tolerance)/step)*step;
    last = floor((limits(2)+tolerance)/step)*step;
    candidate = first:step:last;
    if numel(candidate)>=4 && numel(candidate)<=maxTicks
        candidate(abs(candidate)<100*eps(max(1,max(abs(candidate))))) = 0;
        ticks = candidate;
        return;
    end
end
end


function assert_canvas_fit(fig,stem)
% Catch export regressions before writing an EPS with clipped axes/legend.
axesObjects = findall(fig,'Type','axes');
for k = 1:numel(axesObjects)
    ax = axesObjects(k);
    if strcmpi(ax.Visible,'off'), continue; end
    ax.Units = 'normalized';
    p = ax.Position;
    assert(p(1)>=-0.005 && p(2)>=-0.005 && ...
        p(1)+p(3)<=1.005 && p(2)+p(4)<=1.005, ...
        'Manuscript:AxesOutsideCanvas','Axes outside EPS canvas in %s.',stem);
    lgd = ax.Legend;
    if ~isempty(lgd) && isvalid(lgd)
        lgd.Units = 'normalized'; lp = lgd.Position;
        assert(lp(1)>=-0.005 && lp(2)>=-0.005 && ...
            lp(1)+lp(3)<=1.005 && lp(2)+lp(4)<=1.005, ...
            'Manuscript:LegendOutsideCanvas','Legend outside EPS canvas in %s.',stem);
        assert(lp(2) >= p(2)+p(4)-0.002, ...
            'Manuscript:LegendOverlap','Legend overlaps axes in %s.',stem);
    end
end
end
