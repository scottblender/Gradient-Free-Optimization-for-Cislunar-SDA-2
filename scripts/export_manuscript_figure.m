function metadata = export_manuscript_figure(fig,fileName)
%EXPORT_MANUSCRIPT_FIGURE Export a fully generated manuscript figure.
%
% Styling/layout must already be complete before this function is called.
% This exporter deliberately does not resize fonts, reflow legends, move
% axes, change ticks/cameras, or rewrite the EPS bounding box. It follows the
% stable pre-runner painters export path that produced the correct geometry.

style = reviewer2_paper_style();
[folder,stem,~] = fileparts(char(fileName));
if isempty(folder), folder = pwd; end
if ~isfolder(folder), mkdir(folder); end
base = fullfile(folder,stem);

set(fig,'PaperUnits','inches', ...
    'PaperPositionMode','manual', ...
    'Renderer','painters', ...
    'InvertHardcopy','off', ...
    'Color','w');

paper = double(get(fig,'PaperSize'));
if isempty(paper) || numel(paper)~=2 || any(~isfinite(paper)) || any(paper<=0)
    position = double(get(fig,'Position'));
    paper = position(3:4);
    set(fig,'PaperSize',paper);
end
set(fig,'PaperPosition',[0 0 paper]);

% Finish rendering exactly the figure state produced by the plotter, then
% export it. No figure property is changed after this drawnow.
drawnow;
print(fig,[base '.eps'],'-depsc2','-painters','-r600');
print(fig,[base '.png'],'-dpng',sprintf('-r%d',style.exportDpi));

assert(isfile([base '.eps']) && dir([base '.eps']).bytes>0, ...
    'EPS export failed: %s',base);
assert(isfile([base '.png']) && dir([base '.png']).bytes>0, ...
    'PNG export failed: %s',base);

metadata = struct('stem',stem,'widthInches',paper(1), ...
    'heightInches',paper(2),'minimumFontPoints',style.fontSize, ...
    'placementWidthInches',style.manuscriptPanelWidth, ...
    'minimumPrintedFontPoints', ...
    style.fontSize*style.manuscriptPanelWidth/paper(1));
end
