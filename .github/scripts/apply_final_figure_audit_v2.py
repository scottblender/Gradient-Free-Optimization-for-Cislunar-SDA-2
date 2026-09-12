from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match in {path}; found {count}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


# -----------------------------------------------------------------------------
# Shared trajectory legends: use a compact source canvas sized for one LaTeX
# grid column. The previous 6.5-in-wide legend strip was being reduced to about
# one-third text width, which made otherwise-large legend text appear tiny.
# -----------------------------------------------------------------------------
replace_once(
    'scripts/reviewer2_paper_style.m',
    "style.sharedLegendFigureHeight = 1.05;\n",
    "style.sharedLegendFigureHeight = 1.05;\n"
    "style.sharedGeometryLegendWidth = 3.10;\n"
    "style.sharedGeometryLegendHeight = 1.05;\n",
)

replace_once(
    'scripts/plot_reviewer2_geometry_grid.m',
    "fig = publication_figure(style.figureWidth,style.sharedLegendFigureHeight);",
    "fig = publication_figure(style.sharedGeometryLegendWidth,style.sharedGeometryLegendHeight);",
)

# -----------------------------------------------------------------------------
# Orbit-family plots: explicitly use manuscript case abbreviations and give the
# four optimizer labels substantially more usable horizontal separation. A
# modest 30-degree rotation keeps 20-pt GA/PSO/ABC/ACO labels legible without
# shrinking the global manuscript font.
# -----------------------------------------------------------------------------
replace_once(
    'scripts/make_reviewer2_curated_figures.m',
    'families = ["NHO","SHO","NNRHO","SNRHO","DRO"];\n% Preserve the first mission\'s group ordering and reuse it for all missions.\n',
    'families = ["NHO","SHO","NNRHO","SNRHO","DRO"];\n'
    'caseLabels = ["LG","LT","GI"];\n'
    '% Preserve the first mission\'s group ordering and reuse it for all missions.\n',
)

replace_once(
    'scripts/make_reviewer2_curated_figures.m',
    'withinGroupSpacing = 1.35;\ncaseGap = 2.20;\n',
    'withinGroupSpacing = 1.65;\ncaseGap = 1.60;\n',
)

replace_once(
    'scripts/make_reviewer2_curated_figures.m',
    '    startX = 1 + (m-1)*((nPer-1)*withinGroupSpacing + caseGap + withinGroupSpacing);\n',
    '    startX = 1 + (m-1)*((nPer-1)*withinGroupSpacing + caseGap);\n',
)

replace_once(
    'scripts/make_reviewer2_curated_figures.m',
    "b = bar(ax,x,V,'stacked','BarWidth',0.66); colors = lines(5);",
    "b = bar(ax,x,V,'stacked','BarWidth',0.60); colors = lines(5);",
)

replace_once(
    'scripts/make_reviewer2_curated_figures.m',
    "% space_manuscript_bars applies the general dense-category rotation rule;\n% override it here because the explicit x spacing makes horizontal optimizer\n% labels readable and avoids the previous PSO/ABC/ACO overlap.\n",
    "% Retain the large manuscript tick font and use the shared modest rotation\n"
    "% for dense four-optimizer groups so GA/PSO/ABC/ACO remain distinct.\n",
)

replace_once(
    'scripts/make_reviewer2_curated_figures.m',
    'ax.XTickLabelRotation = 0;\n',
    'ax.XTickLabelRotation = style.categoryLabelAngle;\n',
)

replace_once(
    'scripts/make_reviewer2_curated_figures.m',
    '    text(ax,centers(m),106,mission_short_label(missions(m)), ...\n',
    '    text(ax,centers(m),106,caseLabels(m), ...\n',
)

# -----------------------------------------------------------------------------
# Static configuration audit for the final paper figure requirements.
# -----------------------------------------------------------------------------
replace_once(
    'tests/test_reviewer2_paper_figures_configuration.m',
    "curatedText = string(fileread(fullfile(projectDir,'scripts','make_reviewer2_curated_figures.m')));\n",
    "curatedText = string(fileread(fullfile(projectDir,'scripts','make_reviewer2_curated_figures.m')));\n"
    "geometryGridText = string(fileread(fullfile(projectDir,'scripts','plot_reviewer2_geometry_grid.m')));\n",
)

replace_once(
    'tests/test_reviewer2_paper_figures_configuration.m',
    '    "style.sharedLegendFigureHeight = 1.05", ...\n',
    '    "style.sharedLegendFigureHeight = 1.05", ...\n'
    '    "style.sharedGeometryLegendWidth = 3.10", ...\n'
    '    "style.sharedGeometryLegendHeight = 1.05", ...\n',
)

replace_once(
    'tests/test_reviewer2_paper_figures_configuration.m',
    "assert(contains(styleText,'style.sharedLegendFontSize = 22') && ...\n    contains(runnerText,'SharedLegendStem'), ...\n    'Dense manuscript grids must export and consolidate shared legend strips.');\n",
    "assert(contains(styleText,'style.sharedLegendFontSize = 22') && ...\n"
    "    contains(styleText,'style.sharedGeometryLegendWidth = 3.10') && ...\n"
    "    contains(geometryGridText,'publication_figure(style.sharedGeometryLegendWidth,style.sharedGeometryLegendHeight)') && ...\n"
    "    contains(runnerText,'SharedLegendStem'), ...\n"
    "    'Dense manuscript grids must export compact, readable shared legend strips.');\n",
)

replace_once(
    'tests/test_reviewer2_paper_figures_configuration.m',
    "assert(contains(curatedText,'withinGroupSpacing = 1.35') && ...\n    contains(curatedText,'caseGap = 2.20') && ...\n    contains(curatedText,'ax.XTickLabelRotation = 0'), ...\n    'Dense family-comparison optimizer labels need explicit horizontal spacing.');\n",
    "assert(contains(curatedText,'withinGroupSpacing = 1.65') && ...\n"
    "    contains(curatedText,'caseGap = 1.60') && ...\n"
    "    contains(curatedText,'caseLabels = [\"LG\",\"LT\",\"GI\"]') && ...\n"
    "    contains(curatedText,'ax.XTickLabelRotation = style.categoryLabelAngle'), ...\n"
    "    'Dense family-comparison labels must use LG/LT/GI and readable optimizer spacing.');\n",
)

print('Final figure audit v2 patch applied.')
