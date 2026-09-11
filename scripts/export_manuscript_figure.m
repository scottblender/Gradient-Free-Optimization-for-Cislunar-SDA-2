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
fontObjects = findall(fig,'-property','FontSize');
for k = 1:numel(fontObjects)
    obj = fontObjects(k);
    if isprop(obj,'FontUnits'), obj.FontUnits = 'points'; end
    obj.FontSize = max(obj.FontSize,style.fontSize);
    if isprop(obj,'FontName'), obj.FontName = style.fontName; end
end
% Fit layout only after the final font sizes have been applied.
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
