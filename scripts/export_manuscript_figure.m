function metadata = export_manuscript_figure(fig,fileName)
%EXPORT_MANUSCRIPT_FIGURE Write the completed figure without reformatting it.
% Legacy plotters retain their September 10 EPS print calls. This helper is
% for newer plots only: no font, axes, legend, camera, clipping, or paper edits.
style=reviewer2_paper_style();
[folder,stem,~]=fileparts(char(fileName));
if isempty(folder), folder=pwd; end
if ~isfolder(folder), mkdir(folder); end
base=fullfile(folder,stem);
drawnow;
finalize_manuscript_figure(fig);
print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
print(fig,[base '.png'],'-dpng',sprintf('-r%d',style.exportDpi));
assert(isfile([base '.eps']) && dir([base '.eps']).bytes>0,'EPS export failed: %s',base);
assert(isfile([base '.png']) && dir([base '.png']).bytes>0,'PNG export failed: %s',base);
paper=fig.PaperSize;
metadata=struct('stem',stem,'widthInches',paper(1),'heightInches',paper(2));
end
