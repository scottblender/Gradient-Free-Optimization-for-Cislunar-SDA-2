from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected one match in {path}, found {n}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))

path = 'scripts/make_reviewer2_curated_figures.m'

old = '''% Dense optimizer labels are spaced explicitly rather than rotated or shrunk.
% This keeps GA/PSO/ABC/ACO legible at manuscript scale while retaining clear
% visual separation between the LG, LT, and GI target-case groups.
withinGroupSpacing = 1.65;
caseGap = 1.60;
x = [];
V = [];
tickLabels = strings(0,1);
centers = zeros(3,1);
for m = 1:3
    startX = 1 + (m-1)*((nPer-1)*withinGroupSpacing + caseGap);
'''
new = '''% Use separate spacing rules for the baseline observer-count plot and the
% denser four-optimizer comparison. The latter needs substantially more
% horizontal room for GA/PSO/ABC/ACO at the large manuscript tick size.
isOptimizerPlot = strcmpi(string(groupAxisLabel),"Optimizer");
if isOptimizerPlot
    withinGroupSpacing = 1.90;
    caseGap = 2.80;
    barWidth = 0.54;
    tickRotation = 45;
else
    withinGroupSpacing = 1.35;
    caseGap = 2.50;
    barWidth = 0.60;
    tickRotation = 0;
end
x = [];
V = [];
tickLabels = strings(0,1);
centers = zeros(3,1);
for m = 1:3
    startX = 1 + (m-1)*((nPer-1)*withinGroupSpacing + caseGap);
'''
replace_once(path, old, new)

replace_once(path,
    "b = bar(ax,x,V,'stacked','BarWidth',0.60); colors = lines(5);",
    "b = bar(ax,x,V,'stacked','BarWidth',barWidth); colors = lines(5);")

old = '''% Retain the large manuscript tick font and use the shared modest rotation
% for dense four-optimizer groups so GA/PSO/ABC/ACO remain distinct.
ax.XTick = x;
ax.XTickLabel = cellstr(tickLabels);
ax.XTickLabelRotation = style.categoryLabelAngle;
xlim(ax,[min(x)-0.80*withinGroupSpacing,max(x)+0.80*withinGroupSpacing]);
ylim(ax,[0 112]);
for m = 1:3
    text(ax,centers(m),106,caseLabels(m), ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold');
end
'''
new = '''% Re-apply the final tick labels after style_axes so no generic bar-axis
% formatting can overwrite the orbit-family layout. Optimizer names use a
% fixed 45-degree angle; observer-count labels remain horizontal.
ax.XTick = x;
ax.XTickLabel = cellstr(tickLabels);
ax.XTickLabelRotation = tickRotation;
xlim(ax,[min(x)-0.75*withinGroupSpacing,max(x)+0.75*withinGroupSpacing]);
ylim(ax,[0 116]);

% Use explicit manuscript abbreviations here rather than the general mission
% label helper. This guarantees LG/LT/GI even if other figures retain the
% expanded target names.
for m = 1:3
    text(ax,centers(m),108,char(caseLabels(m)), ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
        'Interpreter','none');
end
'''
replace_once(path, old, new)

# Strengthen static audit so a future formatting pass cannot regress these plots.
test = 'tests/test_reviewer2_paper_figures_configuration.m'
old = '''assert(contains(curatedText,'withinGroupSpacing = 1.65') && ...
    contains(curatedText,'caseGap = 1.60') && ...
    contains(curatedText,'caseLabels = ["LG","LT","GI"]') && ...
    contains(curatedText,'ax.XTickLabelRotation = style.categoryLabelAngle'), ...
    'Dense family-comparison labels must use LG/LT/GI and readable optimizer spacing.');
'''
new = '''assert(contains(curatedText,'withinGroupSpacing = 1.90') && ...
    contains(curatedText,'caseGap = 2.80') && ...
    contains(curatedText,'tickRotation = 45') && ...
    contains(curatedText,'caseLabels = ["LG","LT","GI"]') && ...
    contains(curatedText,'text(ax,centers(m),108,char(caseLabels(m))'), ...
    'Dense family-comparison labels must use explicit LG/LT/GI abbreviations and non-overlapping optimizer spacing.');
'''
replace_once(test, old, new)

print('Applied final orbit-family label and spacing correction.')
