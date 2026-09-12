from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected one match in {path}, found {n}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))

# Give shared trajectory legends enough vertical room for three rows while
# retaining the large 22-pt manuscript legend font.
replace_once(
    'scripts/reviewer2_paper_style.m',
    'style.sharedGeometryLegendWidth = 3.10;\nstyle.sharedGeometryLegendHeight = 1.05;\n',
    'style.sharedGeometryLegendWidth = 3.10;\nstyle.sharedGeometryLegendHeight = 1.55;\n',
)

# The generic manuscript legend policy is intentionally two rows, but these
# compact one-column trajectory legends need up to three rows so their text is
# not clipped or shrunk when placed above the LG/LT/GI manuscript columns.
replace_once(
    'scripts/plot_reviewer2_geometry_grid.m',
    'lgd.ItemTokenSize = style.geometryLegendItemTokenSize;\nlgd.NumColumns = manuscript_legend_columns(labels,style);\nlgd.Units = \'normalized\';\n',
    '''lgd.ItemTokenSize = style.geometryLegendItemTokenSize;\nif panel.mission == "LOW_THRUST_TRANSFER"\n    lgd.NumColumns = 3; % 8 entries -> 3 rows\nelseif panel.mission == "GATEWAY_IMPULSE"\n    lgd.NumColumns = 2; % 6 entries -> 3 rows\nelse\n    lgd.NumColumns = 2; % 5 entries -> 3 rows\nend\nlgd.Units = 'normalized';\n''',
)

# Update the static manuscript-figure audit so future formatting passes retain
# the taller shared-legend canvas and the allowed three-row geometry legends.
replace_once(
    'tests/test_reviewer2_paper_figures_configuration.m',
    '    "style.sharedGeometryLegendHeight = 1.05", ...\n',
    '    "style.sharedGeometryLegendHeight = 1.55", ...\n',
)

replace_once(
    'tests/test_reviewer2_paper_figures_configuration.m',
    "assert(contains(styleText,'style.sharedLegendFontSize = 22') && ...\n    contains(styleText,'style.sharedGeometryLegendWidth = 3.10') && ...\n    contains(geometryGridText,'publication_figure(style.sharedGeometryLegendWidth,style.sharedGeometryLegendHeight)') && ...\n    contains(runnerText,'SharedLegendStem'), ...\n    'Dense manuscript grids must export compact, readable shared legend strips.');\n",
    "assert(contains(styleText,'style.sharedLegendFontSize = 22') && ...\n    contains(styleText,'style.sharedGeometryLegendWidth = 3.10') && ...\n    contains(styleText,'style.sharedGeometryLegendHeight = 1.55') && ...\n    contains(geometryGridText,'publication_figure(style.sharedGeometryLegendWidth,style.sharedGeometryLegendHeight)') && ...\n    contains(geometryGridText,'lgd.NumColumns = 3; % 8 entries -> 3 rows') && ...\n    contains(geometryGridText,'lgd.NumColumns = 2; % 6 entries -> 3 rows') && ...\n    contains(runnerText,'SharedLegendStem'), ...\n    'Dense manuscript grids must export compact, readable shared legends with up to three rows.');\n",
)

print('Applied three-row shared trajectory legend layout.')
