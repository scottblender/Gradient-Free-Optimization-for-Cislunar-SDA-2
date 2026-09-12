from pathlib import Path
import re

# Shared legend typography: larger standalone legends and a dedicated
# full-width shared-legend size for dense manuscript grids.
path = Path('scripts/reviewer2_paper_style.m')
text = path.read_text()
text = text.replace('style.legendFontSize = 16;', 'style.legendFontSize = 20;')
text = text.replace('style.geometryLegendFontSize = 18;', 'style.geometryLegendFontSize = 21;')
text = text.replace('style.legendMinFontSize = 14;', 'style.legendMinFontSize = 17;')
text = text.replace('style.legendItemTokenSize = [18 8];', 'style.legendItemTokenSize = [22 9];')
text = text.replace('style.geometryLegendItemTokenSize = [16 8];', 'style.geometryLegendItemTokenSize = [20 9];')
marker = 'style.legendMinFontSize = 17;\n'
if marker not in text:
    raise RuntimeError('Could not locate legend minimum font marker.')
text = text.replace(marker, marker + 'style.sharedLegendFontSize = 22;\nstyle.sharedLegendFigureHeight = 1.05;\n', 1)
marker = 'style.geometryPlotPosition = [0.10 0.14 0.80 0.70];\n'
if marker not in text:
    raise RuntimeError('Could not locate geometry plot position.')
text = text.replace(marker, marker + 'style.geometryGridPlotPosition = [0.09 0.10 0.82 0.82];\n', 1)
path.write_text(text)

# Dense orbit-family comparison: explicit horizontal spacing between optimizer
# groups, larger mission gaps, and horizontal tick labels.
path = Path('scripts/make_reviewer2_curated_figures.m')
text = path.read_text()
pattern = re.compile(r'function plot_family_grouped_all_cases\(T,groupAxisLabel,stem,out,saveFigures,style\).*?(?=\nfunction plot_objective_family_by_mission)', re.S)
match = pattern.search(text)
if not match:
    raise RuntimeError('Could not locate plot_family_grouped_all_cases.')
new_func = r'''function plot_family_grouped_all_cases(T,groupAxisLabel,stem,out,saveFigures,style)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
families = ["NHO","SHO","NNRHO","SNRHO","DRO"];
% Preserve the first mission's group ordering and reuse it for all missions.
first = T(T.Mission == missions(1),:);
groupKeys = unique(first.GroupKey,'stable');
nPer = numel(groupKeys);

% Dense optimizer labels are spaced explicitly rather than rotated or shrunk.
% This keeps GA/PSO/ABC/ACO legible at manuscript scale while retaining clear
% visual separation between the LG, LT, and GI target-case groups.
withinGroupSpacing = 1.35;
caseGap = 2.20;
x = [];
V = [];
tickLabels = strings(0,1);
centers = zeros(3,1);
for m = 1:3
    startX = 1 + (m-1)*((nPer-1)*withinGroupSpacing + caseGap + withinGroupSpacing);
    xs = startX + (0:nPer-1)*withinGroupSpacing;
    centers(m) = mean(xs);
    x = [x xs];
    for g = 1:nPer
        rows = T(T.Mission == missions(m) & T.GroupKey == groupKeys(g),:);
        assert(height(rows) == 5,'Family summary must contain all five families.');
        values = zeros(1,5);
        for f = 1:5
            row = rows(rows.Family == families(f),:);
            assert(height(row) == 1,'Missing family-selection fraction.');
            values(f) = row.Fraction;
        end
        V(end+1,:) = 100*values;
        tickLabels(end+1,1) = rows.GroupLabel(1);
    end
end

fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
b = bar(ax,x,V,'stacked','BarWidth',0.66); colors = lines(5);
for f = 1:5, b(f).FaceColor = colors(f,:); end
ax.XTick = x;
ax.XTickLabel = cellstr(tickLabels);
xlabel(ax,groupAxisLabel,'FontWeight','bold');
ylabel(ax,'Observer selections (%)','FontWeight','bold');
style_axes(ax,style);

% space_manuscript_bars applies the general dense-category rotation rule;
% override it here because the explicit x spacing makes horizontal optimizer
% labels readable and avoids the previous PSO/ABC/ACO overlap.
ax.XTick = x;
ax.XTickLabel = cellstr(tickLabels);
ax.XTickLabelRotation = 0;
xlim(ax,[min(x)-0.80*withinGroupSpacing,max(x)+0.80*withinGroupSpacing]);
ylim(ax,[0 112]);
for m = 1:3
    text(ax,centers(m),106,mission_short_label(missions(m)), ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold');
end

lgd = legend(ax,b,cellstr(families),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',5,'Box','off');
style_legend(lgd,ax,style);
export_figure(fig,out,stem,saveFigures,style);
end
'''
text = text[:match.start()] + new_func + text[match.end():]
text = text.replace("lgd.FontName = style.fontName; lgd.FontSize = style.fontSize; lgd.FontWeight = 'bold';", "lgd.FontName = style.fontName; lgd.FontSize = style.legendFontSize; lgd.FontWeight = 'bold';")
path.write_text(text)

# Geometry grids: remove repeated per-panel legends and export one full-width
# shared legend for each mission column.
path = Path('scripts/plot_reviewer2_geometry_grid.m')
text = path.read_text()
text = text.replace('    plotPosition = style.geometryPlotPosition;\n', '    plotPosition = style.geometryGridPlotPosition;\n', 1)
old = '''    [legendHandles,legendLabels] = render_geometry_panel(ax,panel,style);\n    limits = equal_span_geometry_limits(common_geometry_limits(panel.allPoints,style));\n'''
new = '''    render_geometry_panel(ax,panel,style);\n    limits = equal_span_geometry_limits(common_geometry_limits(panel.allPoints,style));\n'''
if old not in text:
    raise RuntimeError('Could not locate geometry render/legend handle block.')
text = text.replace(old,new,1)
old = '''    legendHandle = legend(ax,legendHandles,cellstr(legendLabels), ...\n        'Location','northoutside','Orientation','horizontal');\n    format_case_legend(legendHandle,panel.mission,style);\n    center_reference_legend(ax,legendHandle,plotPosition,style);\n\n'''
if old not in text:
    raise RuntimeError('Could not locate repeated geometry legend block.')
text = text.replace(old,'',1)
marker = '''for k = 1:n\n    mission = string(selection.Mission(k));\n    panel = panelCells{k};\n'''
if marker not in text:
    raise RuntimeError('Could not locate geometry panel loop.')
shared_setup = '''selectionMissions = string(selection.Mission);\nsharedLegendStem = strings(n,1);\nlegendMissions = unique(selectionMissions,'stable');\nfor mission = legendMissions(:)'\n    idx = find(selectionMissions == mission,1,'first');\n    legendStem = stemPrefix + "_legend_" + mission_code(mission);\n    export_shared_geometry_legend(panelCells{idx},figureDir,legendStem,saveFigures,style);\n    sharedLegendStem(selectionMissions == mission) = legendStem;\nend\n\n'''
text = text.replace(marker, shared_setup + marker, 1)
old = '''    numObservers,families,orbitIndices,slotIndices,figureStem,figureStem, ...\n    'VariableNames',{'Mission','PanelKey','PanelLabel','RepresentativeObjective', ...\n    'GroupMeanObjective','GroupStdObjective','RepresentativeSeed','RunFile', ...\n    'NumObservers','OrbitFamilies','OrbitIndices','SlotIndices', ...\n    'FigureStem','GridFigureStem'});\n'''
new = '''    numObservers,families,orbitIndices,slotIndices,figureStem,figureStem,sharedLegendStem, ...\n    'VariableNames',{'Mission','PanelKey','PanelLabel','RepresentativeObjective', ...\n    'GroupMeanObjective','GroupStdObjective','RepresentativeSeed','RunFile', ...\n    'NumObservers','OrbitFamilies','OrbitIndices','SlotIndices', ...\n    'FigureStem','GridFigureStem','SharedLegendStem'});\n'''
if old not in text:
    raise RuntimeError('Could not locate geometry details table.')
text = text.replace(old,new,1)
insert_before = '\n\nfunction format_case_legend(lgd,mission,style)\n'
if insert_before not in text:
    raise RuntimeError('Could not locate geometry helper insertion point.')
helper = r'''

function export_shared_geometry_legend(panel,figureDir,stem,saveFigures,style)
if ~saveFigures, return; end
fig = publication_figure(style.figureWidth,style.sharedLegendFigureHeight);
ax = axes(fig,'Units','normalized','Position',[0.01 0.01 0.98 0.98], ...
    'Visible','off');
hold(ax,'on');
obsColor = lines(1);
obsColor = obsColor(1,:);
moonColor = [0.72 0.72 0.72];
pointColor = [0.80 0.80 0.80];

if panel.mission == "LOW_THRUST_TRANSFER"
    handles = [ ...
        plot(ax,nan,nan,'-','Color',[0.70 0.70 0.70],'LineWidth',1.4), ...
        plot(ax,nan,nan,'-','Color',panel.targetColor,'LineWidth',2.8), ...
        plot(ax,nan,nan,'-','Color',obsColor,'LineWidth',1.8), ...
        plot(ax,nan,nan,'o','MarkerSize',9,'MarkerFaceColor',reviewer2_target_color("LUNAR_GATEWAY"),'MarkerEdgeColor','k'), ...
        plot(ax,nan,nan,'s','MarkerSize',9,'MarkerFaceColor',panel.targetColor,'MarkerEdgeColor','k'), ...
        plot(ax,nan,nan,'s','MarkerSize',9,'MarkerFaceColor',moonColor,'MarkerEdgeColor','none'), ...
        plot(ax,nan,nan,'^','MarkerSize',9,'MarkerFaceColor',pointColor,'MarkerEdgeColor','k'), ...
        plot(ax,nan,nan,'v','MarkerSize',9,'MarkerFaceColor',pointColor,'MarkerEdgeColor','k')];
    labels = ["Endpt.","LT","Obs.","Start","End","Moon","L1","L2"];
elseif panel.mission == "GATEWAY_IMPULSE"
    gatewayColor = reviewer2_target_color("LUNAR_GATEWAY");
    nominalColor = 0.45*gatewayColor + 0.55*[1 1 1];
    handles = [ ...
        plot(ax,nan,nan,'--','Color',nominalColor,'LineWidth',2.2), ...
        plot(ax,nan,nan,'-','Color',panel.targetColor,'LineWidth',2.8), ...
        plot(ax,nan,nan,'-','Color',obsColor,'LineWidth',1.8), ...
        plot(ax,nan,nan,'s','MarkerSize',9,'MarkerFaceColor',moonColor,'MarkerEdgeColor','none'), ...
        plot(ax,nan,nan,'^','MarkerSize',9,'MarkerFaceColor',pointColor,'MarkerEdgeColor','k'), ...
        plot(ax,nan,nan,'v','MarkerSize',9,'MarkerFaceColor',pointColor,'MarkerEdgeColor','k')];
    labels = ["Nominal LG","GI","Obs.","Moon","L1","L2"];
else
    handles = [ ...
        plot(ax,nan,nan,'-','Color',panel.targetColor,'LineWidth',2.8), ...
        plot(ax,nan,nan,'-','Color',obsColor,'LineWidth',1.8), ...
        plot(ax,nan,nan,'s','MarkerSize',9,'MarkerFaceColor',moonColor,'MarkerEdgeColor','none'), ...
        plot(ax,nan,nan,'^','MarkerSize',9,'MarkerFaceColor',pointColor,'MarkerEdgeColor','k'), ...
        plot(ax,nan,nan,'v','MarkerSize',9,'MarkerFaceColor',pointColor,'MarkerEdgeColor','k')];
    labels = ["LG","Obs.","Moon","L1","L2"];
end

lgd = legend(ax,handles,cellstr(labels),'Location','none', ...
    'Orientation','horizontal','Box','off');
lgd.FontName = style.fontName;
lgd.FontSize = style.sharedLegendFontSize;
lgd.FontWeight = style.fontWeight;
lgd.ItemTokenSize = style.geometryLegendItemTokenSize;
lgd.NumColumns = manuscript_legend_columns(labels,style);
lgd.Units = 'normalized';
drawnow;
pos = lgd.Position;
pos(1) = 0.5-pos(3)/2;
pos(2) = 0.5-pos(4)/2;
lgd.Position = pos;
lgd.AutoUpdate = 'off';
axis(ax,'off');
drawnow;
base = fullfile(char(figureDir),char(stem));
print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
close(fig);
end
'''
text = text.replace(insert_before, helper + insert_before, 1)
path.write_text(text)

# Monte Carlo grids: remove repeated per-panel legends and produce one large
# shared legend strip for the LG and LT/GI grids.
path = Path('scripts/plot_reviewer2_baseline_monte_carlo.m')
text = path.read_text()
marker = 'figureStem = strings(height(summary),1);\n'
if marker not in text:
    raise RuntimeError('Could not locate MC figureStem marker.')
text = text.replace(marker, marker + 'sharedLegendStem = "baseline_mc_shared_legend";\nexport_mc_shared_legend(figureDir,sharedLegendStem,saveFigures,style);\n', 1)
old = '''    lgd = legend(ax,[hBox hRef],{'Monte Carlo samples','Optimized reference'}, ...\n        'Location','northoutside','Orientation','horizontal','Box','off');\n    lgd.FontName = style.fontName;\n    lgd.FontSize = style.fontSize;\n    lgd.FontWeight = 'bold';\n    format_manuscript_legend(ax,lgd,style,style.metricPlotPosition);\n\n'''
if old not in text:
    raise RuntimeError('Could not locate repeated MC legend block.')
text = text.replace(old,'',1)
old = 'details.FigureDirectory = repmat(figureDir,height(details),1);\nend\n'
new = 'details.FigureDirectory = repmat(figureDir,height(details),1);\ndetails.SharedLegendStem = repmat(sharedLegendStem,height(details),1);\nend\n'
if old not in text:
    raise RuntimeError('Could not locate MC details table ending.')
text = text.replace(old,new,1)
insert_before = '\n\nfunction enforce_minimum_font_size(fig,minFontSize)\n'
if insert_before not in text:
    raise RuntimeError('Could not locate MC helper insertion point.')
helper = r'''

function export_mc_shared_legend(figureDir,stem,saveFigures,style)
if ~saveFigures, return; end
widthIn = style.figureWidth;
heightIn = style.sharedLegendFigureHeight;
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
ax = axes(fig,'Units','normalized','Position',[0.01 0.01 0.98 0.98], ...
    'Visible','off');
hold(ax,'on');
hSamples = plot(ax,nan,nan,'s','LineStyle','none','MarkerSize',12, ...
    'MarkerFaceColor',style.optimizerColors(1,:), ...
    'MarkerEdgeColor',style.optimizerColors(1,:));
hReference = plot(ax,nan,nan,'-','Color',[1.00 0.30 0.30],'LineWidth',1.8);
labels = ["Monte Carlo samples","Optimized reference"];
lgd = legend(ax,[hSamples hReference],cellstr(labels),'Location','none', ...
    'Orientation','horizontal','NumColumns',2,'Box','off');
lgd.FontName = style.fontName;
lgd.FontSize = style.sharedLegendFontSize;
lgd.FontWeight = style.fontWeight;
lgd.ItemTokenSize = style.legendItemTokenSize;
lgd.Units = 'normalized';
drawnow;
pos = lgd.Position;
pos(1) = 0.5-pos(3)/2;
pos(2) = 0.5-pos(4)/2;
lgd.Position = pos;
lgd.AutoUpdate = 'off';
axis(ax,'off');
drawnow;
base = fullfile(char(figureDir),char(stem));
print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
close(fig);
end
'''
text = text.replace(insert_before, helper + insert_before, 1)
path.write_text(text)

# Root figure runner must consolidate the new shared MC legend. Geometry legend
# strips are automatically collected with result EPS files.
path = Path('run_manuscript_figures.m')
text = path.read_text()
old = '        sources = [sources;fullfile(details.FigureDirectory,details.FigureStem+".eps")];\n'
new = '''        sources = [sources;fullfile(details.FigureDirectory,details.FigureStem+".eps")];
        if ismember('SharedLegendStem',details.Properties.VariableNames)
            sharedStems = unique(string(details.SharedLegendStem),'stable');
            sharedStems = sharedStems(strlength(sharedStems) > 0);
            if ~isempty(sharedStems)
                sources = [sources;fullfile(details.FigureDirectory(1),sharedStems+".eps")];
            end
        end
'''
if old not in text:
    raise RuntimeError('Could not locate MC consolidation line.')
text = text.replace(old,new,1)
path.write_text(text)

# Configuration audit: lock in the larger legends, shared grid legend assets,
# and explicit family-chart spacing.
path = Path('tests/test_reviewer2_paper_figures_configuration.m')
text = path.read_text()
text = text.replace('"style.legendFontSize = 16", ...', '"style.legendFontSize = 20", ...')
text = text.replace('"style.geometryLegendFontSize = 18", ...', '"style.geometryLegendFontSize = 21", ...')
marker = '    "style.geometryLegendFontSize = 21", ...\n'
if marker not in text:
    raise RuntimeError('Could not locate test legend-font marker.')
text = text.replace(marker, marker + '    "style.sharedLegendFontSize = 22", ...\n    "style.sharedLegendFigureHeight = 1.05", ...\n', 1)
marker = '    "style.geometryPlotPosition = [0.10 0.14 0.80 0.70]", ...\n'
if marker not in text:
    raise RuntimeError('Could not locate test geometry position marker.')
text = text.replace(marker, marker + '    "style.geometryGridPlotPosition = [0.09 0.10 0.82 0.82]", ...\n', 1)
old = '''assert(contains(barText,'numel(labels) >= 8') && ...
    contains(barText,'style.categoryLabelAngle'), ...
    'Dense family-comparison optimizer labels need explicit rotation.');
'''
new = '''assert(contains(curatedText,'withinGroupSpacing = 1.35') && ...
    contains(curatedText,'caseGap = 2.20') && ...
    contains(curatedText,'ax.XTickLabelRotation = 0'), ...
    'Dense family-comparison optimizer labels need explicit horizontal spacing.');
'''
if old not in text:
    raise RuntimeError('Could not locate old family-label test.')
text = text.replace(old,new,1)
marker = '''assert(contains(parallelText,'style.figureWidth style.figureHeight') && ...
    contains(parallelText,'format_manuscript_legend'), ...
    'Parallel figures must use the common canvas and legend formatter.');
'''
shared_test = marker + '''assert(contains(styleText,'style.sharedLegendFontSize = 22') && ...
    contains(runnerText,'SharedLegendStem'), ...
    'Dense manuscript grids must export and consolidate shared legend strips.');
'''
if marker not in text:
    raise RuntimeError('Could not locate legend audit insertion point.')
text = text.replace(marker,shared_test,1)
path.write_text(text)

# Basic static checks before committing.
checks = {
    'scripts/reviewer2_paper_style.m': ['style.legendFontSize = 20','style.sharedLegendFontSize = 22'],
    'scripts/make_reviewer2_curated_figures.m': ['withinGroupSpacing = 1.35','ax.XTickLabelRotation = 0'],
    'scripts/plot_reviewer2_geometry_grid.m': ['export_shared_geometry_legend','SharedLegendStem'],
    'scripts/plot_reviewer2_baseline_monte_carlo.m': ['export_mc_shared_legend','SharedLegendStem'],
    'run_manuscript_figures.m': ['sharedStems = unique(string(details.SharedLegendStem)']
}
for file, tokens in checks.items():
    t = Path(file).read_text()
    for token in tokens:
        if token not in t:
            raise RuntimeError(f'Missing expected token {token!r} in {file}')
