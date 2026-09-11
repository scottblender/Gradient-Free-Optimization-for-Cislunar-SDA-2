function metadata = export_manuscript_figure(fig,fileName)
%EXPORT_MANUSCRIPT_FIGURE Export an already-final manuscript figure.
%
% Required workflow:
%   style = reviewer2_paper_style();
%   generate the complete figure using style;
%   export_manuscript_figure(fig,fileName);
%
% This function deliberately does NOT change fonts, ticks, axes positions,
% legends, cameras, limits, plotted data, or object styling. In particular,
% 3-D graphics are never mutated immediately before the painters EPS pass.

style = reviewer2_paper_style();
[folder,stem,~] = fileparts(char(fileName));
if isempty(folder), folder = pwd; end
if ~isfolder(folder), mkdir(folder); end
base = fullfile(folder,stem);
finalEps = [base '.eps'];
finalPng = [base '.png'];

% Export configuration only. Paper geometry is part of the output format,
% not a second figure-layout pass.
set(fig,'PaperUnits','inches','PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off','Color','w');
paper = double(get(fig,'PaperSize'));
set(fig,'PaperPosition',[0 0 paper]);

% Validate the figure exactly as the plotter finished it. These checks are
% read-only and intentionally occur before either file is written.
assert_canvas_fit(fig,stem);
assert_manuscript_fonts(fig,style,stem);
assert_geometry_has_content(fig,stem);
assert_eps_compatible_objects(fig,stem);

drawnow;

% Do not leave stale exports behind if generation/export fails.
if isfile(finalEps), delete(finalEps); end
if isfile(finalPng), delete(finalPng); end

tempBase = tempname(folder);
tempEps = [tempBase '.eps'];
tempPng = [tempBase '.png'];
tempCleanup = onCleanup(@() cleanup_temporary_exports(tempEps,tempPng));

% Use the stable runner-era EPS path. No styling/layout operation occurs
% between the final drawnow and these print calls.
print(fig,tempEps,'-depsc2','-painters','-loose');
print(fig,tempPng,'-dpng',sprintf('-r%d',style.exportDpi));
assert(isfile(tempEps) && dir(tempEps).bytes>0,'EPS export failed: %s',base);
assert(isfile(tempPng) && dir(tempPng).bytes>0,'PNG export failed: %s',base);

% Normalize DSC bounds to the declared paper rectangle. This edits only EPS
% metadata and never rescales/replots the figure contents.
epsText = fileread(tempEps);
assert(startsWith(epsText,'%!PS-Adobe'),'Invalid EPS output: %s',base);
widthPt = 72*paper(1);
heightPt = 72*paper(2);
box = sprintf('%%%%BoundingBox: 0 0 %d %d',ceil(widthPt),ceil(heightPt));
hires = sprintf('%%%%HiResBoundingBox: 0 0 %.6f %.6f',widthPt,heightPt);
epsText = regexprep(epsText,'(?m)^%%BoundingBox:[^\r\n]*',box);
if contains(epsText,'%%HiResBoundingBox:')
    epsText = regexprep(epsText,'(?m)^%%HiResBoundingBox:[^\r\n]*',hires);
else
    firstNewline = find(epsText==newline,1);
    epsText = [epsText(1:firstNewline) hires newline epsText(firstNewline+1:end)];
end
fid = fopen(tempEps,'w');
assert(fid~=-1,'Cannot write EPS: %s',base);
fileCleanup = onCleanup(@() fclose(fid));
fwrite(fid,epsText,'char');
clear fileCleanup;

[ok,message] = movefile(tempEps,finalEps,'f');
assert(ok,'Could not finalize EPS %s: %s',finalEps,message);
[ok,message] = movefile(tempPng,finalPng,'f');
assert(ok,'Could not finalize PNG %s: %s',finalPng,message);
clear tempCleanup;

metadata = struct('stem',stem,'widthInches',paper(1), ...
    'heightInches',paper(2),'minimumFontPoints',style.fontSize, ...
    'placementWidthInches',style.manuscriptPanelWidth, ...
    'minimumPrintedFontPoints',style.fontSize*style.manuscriptPanelWidth/paper(1));
end


function assert_canvas_fit(fig,stem)
% Read-only clipping/overlap guard for the final generated state.
axesObjects = findall(fig,'Type','axes');
for k = 1:numel(axesObjects)
    ax = axesObjects(k);
    if strcmpi(ax.Visible,'off'), continue; end

    p = hgconvertunits(fig,double(ax.Position),ax.Units,'normalized',fig);
    ti = hgconvertunits(fig,double(ax.TightInset),ax.Units,'normalized',fig);
    envelope = [p(1)-ti(1),p(2)-ti(2), ...
        p(1)+p(3)+ti(3),p(2)+p(4)+ti(4)];
    assert(envelope(1)>=-0.01 && envelope(2)>=-0.01 && ...
        envelope(3)<=1.01 && envelope(4)<=1.01, ...
        'Manuscript:TextOutsideCanvas', ...
        'Axes labels/ticks extend outside EPS canvas in %s.',stem);

    lgd = ax.Legend;
    if isempty(lgd) || ~isvalid(lgd), continue; end
    lp = hgconvertunits(fig,double(lgd.Position),lgd.Units,'normalized',fig);
    assert(lp(1)>=-0.005 && lp(2)>=-0.005 && ...
        lp(1)+lp(3)<=1.005 && lp(2)+lp(4)<=1.005, ...
        'Manuscript:LegendOutsideCanvas','Legend outside EPS canvas in %s.',stem);

    % For quantitative 2-D figures, an outside/manual-above legend must sit
    % above the axes/tick envelope rather than covering labels or data.
    if ~is_geometry_axis(ax)
        location = lower(string(lgd.Location));
        isAboveLegend = contains(location,'northoutside') || ...
            (location=="none" && lp(2)>=p(2)+0.5*p(4));
        if isAboveLegend
            assert(lp(2)>=p(2)+p(4)+ti(4)-0.004, ...
                'Manuscript:LegendOverlap', ...
                'Legend overlaps metric axes/tick labels in %s.',stem);
        end
    end
end
end


function assert_manuscript_fonts(fig,style,stem)
objects = findall(fig,'-property','FontSize');
for k = 1:numel(objects)
    obj = objects(k);
    try
        assert(double(obj.FontSize)>=style.fontSize-1e-9, ...
            'Manuscript:FontTooSmall','Font below manuscript size in %s.',stem);
        if isprop(obj,'FontWeight')
            assert(strcmpi(string(obj.FontWeight),style.fontWeight), ...
                'Manuscript:FontNotBold','Non-bold manuscript text in %s.',stem);
        end
    catch err
        if startsWith(err.identifier,'Manuscript:'), rethrow(err); end
    end
end
end


function assert_geometry_has_content(fig,stem)
axesObjects = findall(fig,'Type','axes');
for k = 1:numel(axesObjects)
    ax = axesObjects(k);
    if strcmpi(ax.Visible,'off') || ~is_geometry_axis(ax), continue; end
    drawable = [findall(ax,'Type','line');findall(ax,'Type','surface'); ...
        findall(ax,'Type','patch');findall(ax,'Type','scatter')];
    assert(~isempty(drawable),'Manuscript:EmptyGeometry', ...
        'Geometry axes contain no drawable objects before export in %s.',stem);
end
end


function tf = is_geometry_axis(ax)
labels = lower([label_string(ax.XLabel),label_string(ax.YLabel),label_string(ax.ZLabel)]);
v = view(ax);
tf = any(contains(labels,'(lu)')) || abs(v(1))>1e-9 || abs(v(2)-90)>1e-9;
end


function value = label_string(handle)
value = string(handle.String);
if isempty(value), value=""; else, value=strjoin(value(:).'," "); end
end


function assert_eps_compatible_objects(fig,stem)
for property = ["FaceAlpha","EdgeAlpha"]
    objects = findall(fig,'-property',char(property));
    for k = 1:numel(objects)
        value = get(objects(k),char(property));
        assert(isnumeric(value) && all(value(:)==1), ...
            'Manuscript:Transparency', ...
            'EPS requires opaque %s in %s.',property,stem);
    end
end
end


function cleanup_temporary_exports(epsFile,pngFile)
if isfile(epsFile), delete(epsFile); end
if isfile(pngFile), delete(pngFile); end
end
