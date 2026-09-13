from pathlib import Path

p = Path('scripts/plot_study_definition_figures.m')
text = p.read_text()

old = """figureFiles = strings(numel(familyGroups),1);\n\nfor groupIndex = 1:numel(familyGroups)\n"""
new = """figureFiles = strings(numel(familyGroups),1);\nsharedLegendFile = export_orbit_catalog_family_legend( ...\n    outputDir,cL1,cL2,cMoon,cPoint,style);\n\nfor groupIndex = 1:numel(familyGroups)\n"""
if text.count(old) != 1:
    raise RuntimeError(f'Expected one figureFiles marker, found {text.count(old)}')
text = text.replace(old,new,1)

old = """    if numel(group)==2\n        legendHandle = legend(ax,legendHandles,cellstr(legendLabels), ...\n            'Location','northoutside','Orientation','horizontal');\n        legendColumns = numel(legendLabels);\n    else\n        legendHandle = legend(ax,legendHandles,cellstr(legendLabels), ...\n            'Location','northeast','Orientation','vertical');\n        legendColumns = 1;\n    end\n    format_study_legend(legendHandle,legendColumns,[14 8]);\n\n    if numel(group)==2\n        finalize_centered_3d_axes(ax,legendHandle,plotPosition);\n    else\n        ax.Units = 'normalized';\n        format_manuscript_legend(ax,legendHandle,style,plotPosition);\n        ax.LooseInset = max(ax.TightInset,0.015);\n    end\n"""
new = """    if numel(group)==2\n        % The four paired orbit-family panels share one legend-only EPS.\n        % Keep each data panel legend-free so LaTeX can place the common\n        % legend above the 2x2 subfigure group without shrinking the plots.\n        ax.Units = 'normalized';\n        ax.Position = plotPosition;\n        ax.LooseInset = max(ax.TightInset,0.015);\n    else\n        % DRO is a standalone panel and retains its local legend.\n        legendHandle = legend(ax,legendHandles,cellstr(legendLabels), ...\n            'Location','northeast','Orientation','vertical');\n        format_study_legend(legendHandle,1,[14 8]);\n        ax.Units = 'normalized';\n        format_manuscript_legend(ax,legendHandle,style,plotPosition);\n        ax.LooseInset = max(ax.TightInset,0.015);\n    end\n"""
if text.count(old) != 1:
    raise RuntimeError(f'Expected one embedded family legend block, found {text.count(old)}')
text = text.replace(old,new,1)

old = """outputs.figures = figureFiles;\noutputs.familySummary = string(summaryFile);\n"""
new = """outputs.figures = figureFiles;\noutputs.sharedLegend = string(sharedLegendFile);\noutputs.familySummary = string(summaryFile);\n"""
if text.count(old) != 1:
    raise RuntimeError(f'Expected one outputs marker, found {text.count(old)}')
text = text.replace(old,new,1)

marker = """fprintf('Saved orbit-family figures and catalog tables to:\\n  %s\\n',outputDir);\nend\n\nfunction outputs = create_slot_definition(inspectFigure)\n"""
helper = r'''fprintf('Saved orbit-family figures and catalog tables to:\n  %s\n',outputDir);
end

function legendFile = export_orbit_catalog_family_legend( ...
    outputDir,cL1,cL2,cMoon,cPoint,style)
%EXPORT_ORBIT_CATALOG_FAMILY_LEGEND Shared legend for the four paired family panels.

fig = publication_figure(style.sharedResultLegendWidth, ...
    style.sharedResultLegendHeight);
cleanup = onCleanup(@() close(fig));
ax = axes(fig,'Units','normalized','Position',[0.01 0.01 0.98 0.98], ...
    'Visible','off');
hold(ax,'on');

hL1Family = plot(ax,nan,nan,'-','Color',cL1,'LineWidth',1.4);
hL2Family = plot(ax,nan,nan,'-','Color',cL2,'LineWidth',1.4);
hMoon = plot(ax,nan,nan,'o','LineStyle','none','MarkerSize',7, ...
    'MarkerFaceColor',cMoon,'MarkerEdgeColor',[0.45,0.45,0.45], ...
    'LineWidth',0.9);
hL1Point = plot(ax,nan,nan,'^','LineStyle','none','MarkerSize',8, ...
    'MarkerFaceColor',cPoint,'MarkerEdgeColor',[0.55,0.55,0.55], ...
    'LineWidth',0.9);
hL2Point = plot(ax,nan,nan,'v','LineStyle','none','MarkerSize',8, ...
    'MarkerFaceColor',cPoint,'MarkerEdgeColor',[0.55,0.55,0.55], ...
    'LineWidth',0.9);

lgd = legend(ax,[hL1Family,hL2Family,hMoon,hL1Point,hL2Point], ...
    {'L1','L2','Moon','L1 point','L2 point'}, ...
    'Location','none','Orientation','horizontal','NumColumns',3,'Box','off');
lgd.FontName = style.fontName;
lgd.FontSize = style.sharedLegendFontSize;
lgd.FontWeight = style.fontWeight;
lgd.ItemTokenSize = [14 8];
lgd.Units = 'normalized';
drawnow;
pos = lgd.Position;
pos(1) = 0.5-pos(3)/2;
pos(2) = 0.5-pos(4)/2;
lgd.Position = pos;
lgd.AutoUpdate = 'off';
axis(ax,'off');
drawnow;

legendFile = fullfile(outputDir,'orbit_catalog_family_legend.eps');
print(fig,legendFile,'-depsc2','-painters','-r600','-loose');
print(fig,fullfile(outputDir,'orbit_catalog_family_legend.png'),'-dpng','-r300');
clear cleanup;
end

function outputs = create_slot_definition(inspectFigure)
'''
if text.count(marker) != 1:
    raise RuntimeError(f'Expected one helper insertion marker, found {text.count(marker)}')
text = text.replace(marker,helper,1)

p.write_text(text)
print('Updated study-definition family plots to use a separate shared legend EPS.')
