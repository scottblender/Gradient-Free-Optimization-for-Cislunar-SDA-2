function metadata = export_manuscript_figure(fig,fileName)
%EXPORT_MANUSCRIPT_FIGURE Vector EPS and PNG with identical fixed canvases.
% Keep the complete paper rectangle: tight cropping makes paired diagrams
% scale differently in LaTeX. Font sizes are specified at the export size.
style = reviewer2_paper_style();
[folder,stem,~] = fileparts(char(fileName));
if isempty(folder), folder = pwd; end
if ~isfolder(folder), mkdir(folder); end
base = fullfile(folder,stem);
set(fig,'PaperUnits','inches','PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off','Color','w');
paper = get(fig,'PaperSize');
set(fig,'PaperPosition',[0 0 paper]);

% Shorten repeated mission/case wording before final layout. This keeps the
% scientific terminology intact while using compact plot-facing labels such
% as LG, LT, and GI where the full names only consume figure space.
abbreviate_manuscript_text(fig);

% Repeated optimizer labels in the orbit-family selection summary are dense
% at the final manuscript font. Spread those category centers slightly while
% preserving the larger gaps between mission blocks. This is intentionally
% limited to repeated GA/PSO/ABC/ACO/BO categorical axes.
spread_repeated_optimizer_groups(fig);

fontObjects = findall(fig,'-property','FontSize');
for k = 1:numel(fontObjects)
    obj = fontObjects(k);
    if isprop(obj,'FontUnits'), obj.FontUnits = 'points'; end
    obj.FontSize = max(obj.FontSize,style.fontSize);
    if isprop(obj,'FontName'), obj.FontName = style.fontName; end
    if isprop(obj,'FontWeight'), obj.FontWeight = style.fontWeight; end
end
% Fit layout only after the final font sizes and weights have been applied.
layout_manuscript_figure(fig,style);
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
% MATLAB releases differ in their EPS bounding-box padding. Normalize both
% DSC boxes to the physical paper rectangle, without rescaling the drawing.
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

function abbreviate_manuscript_text(fig)
%ABBREVIATE_MANUSCRIPT_TEXT Compact repeated case names in final figures.
% Apply to legends, categorical tick labels, and free text. Axis variable
% names and mathematical notation are otherwise left unchanged.
legendObjects = findall(fig,'Type','legend');
for k = 1:numel(legendObjects)
    legendObjects(k).String = abbreviate_value(legendObjects(k).String,true);
end

axesObjects = findall(fig,'Type','axes');
for k = 1:numel(axesObjects)
    ax = axesObjects(k);
    abbreviate_tick_labels(ax,'X');
    abbreviate_tick_labels(ax,'Y');
    abbreviate_tick_labels(ax,'Z');
end

textObjects = findall(fig,'Type','text');
for k = 1:numel(textObjects)
    try
        value = textObjects(k).String;
        shortened = abbreviate_value(value,false);
        if ~isequal(value,shortened)
            textObjects(k).String = shortened;
        end
    catch
        % Ignore graphics proxy objects that expose non-writable String data.
    end
end
end

function spread_repeated_optimizer_groups(fig)
%SPREAD_REPEATED_OPTIMIZER_GROUPS Add modest spacing inside mission blocks.
axesObjects = findall(fig,'Type','axes');
allowed = ["GA","PSO","ABC","ACO","BO"];
for k = 1:numel(axesObjects)
    ax = axesObjects(k);
    labels = upper(strip(string(ax.XTickLabel(:))));
    ticks = double(ax.XTick(:));
    if numel(labels) < 8 || numel(labels) ~= numel(ticks) || ...
            any(~ismember(labels,allowed))
        continue;
    end

    repeatIndex = find(labels(2:end) == labels(1),1,'first');
    if isempty(repeatIndex), continue; end
    groupSize = repeatIndex;
    if groupSize < 2 || mod(numel(labels),groupSize) ~= 0, continue; end
    pattern = labels(1:groupSize);
    numGroups = numel(labels)/groupSize;
    validPattern = true;
    for g = 1:numGroups
        idx = (g-1)*groupSize+(1:groupSize);
        if ~isequal(labels(idx),pattern)
            validPattern = false;
            break;
        end
    end
    if ~validPattern, continue; end

    oldTicks = ticks(:).';
    if numel(oldTicks) < 2 || any(diff(oldTicks) <= 0), continue; end
    baseWithin = median(diff(oldTicks(1:groupSize)));
    if ~isfinite(baseWithin) || baseWithin <= 0, continue; end
    withinSpacing = 1.15*baseWithin;
    if numGroups > 1
        oldGap = oldTicks(groupSize+1)-oldTicks(groupSize);
    else
        oldGap = 1.5*withinSpacing;
    end
    groupGap = max(oldGap,1.45*withinSpacing);

    newTicks = zeros(size(oldTicks));
    groupCentersOld = zeros(numGroups,1);
    groupCentersNew = zeros(numGroups,1);
    start = oldTicks(1);
    for g = 1:numGroups
        idx = (g-1)*groupSize+(1:groupSize);
        if g > 1
            start = newTicks(idx(1)-1)+groupGap;
        end
        newTicks(idx) = start+(0:groupSize-1)*withinSpacing;
        groupCentersOld(g) = mean(oldTicks(idx));
        groupCentersNew(g) = mean(newTicks(idx));
    end

    bars = findall(ax,'Type','Bar');
    movedBar = false;
    for b = 1:numel(bars)
        xData = double(bars(b).XData(:).');
        if numel(xData) == numel(oldTicks) && ...
                max(abs(xData-oldTicks)) <= 100*eps(max(1,max(abs(oldTicks))))
            bars(b).XData = newTicks;
            movedBar = true;
        end
    end
    if ~movedBar, continue; end

    ax.XTick = newTicks;
    textObjects = findall(ax,'Type','text');
    missionLabels = ["LG","LT","GI"];
    missionIndex = 0;
    for t = 1:numel(textObjects)
        value = string(textObjects(t).String);
        if isscalar(value) && any(value == missionLabels)
            missionIndex = missionIndex+1;
            if missionIndex <= numGroups
                pos = textObjects(t).Position;
                [~,nearest] = min(abs(groupCentersOld-pos(1)));
                pos(1) = groupCentersNew(nearest);
                textObjects(t).Position = pos;
            end
        end
    end
    drawnow;
end
end

function abbreviate_tick_labels(ax,axisName)
property = [axisName 'TickLabel'];
value = ax.(property);
if isempty(value), return; end
shortened = abbreviate_value(value,false);
% Do not touch ordinary numeric labels. Assigning an unchanged TickLabel
% would switch some MATLAB axes from automatic to manual label management.
if ~isequal(value,shortened)
    ax.(property) = shortened;
end
end

function output = abbreviate_value(value,isLegend)
if ~(ischar(value) || isstring(value) || iscell(value))
    output = value;
    return;
end

wasChar = ischar(value);
wasCell = iscell(value);
text = string(value);

% Replace the full case names first, then common shorter references.
text = replace(text,"Lunar Gateway","LG");
text = replace(text,"Low-thrust transfer","LT");
text = replace(text,"Gateway impulse","GI");
text = replace(text,"Gateway-impulse","GI");
text = replace(text,"Gateway","LG");
text = replace(text,"Low-thrust","LT");

if isLegend
    % Geometry legends repeat these descriptions across many panels. Keep
    % the meaning obvious while conserving enough horizontal space for the
    % final 22-point manuscript font.
    text(text=="Post-impulse") = "GI traj.";
    text(text=="Transfer") = "LT traj.";
    text(text=="Target trajectory") = "Target";
    text(text=="Observer orbits") = "Obs. orbits";
    text(text=="Endpoint orbits") = "Endpoints";
    text(text=="6000-FE GA reference") = "6000-FE GA ref.";
end

if wasChar
    output = char(text);
elseif wasCell
    output = cellstr(text);
else
    output = text;
end
end
