from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected one match in {path}, found {n}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1))

# Give baseline and comparison orbit-family groups unique legend stems so a
# full manuscript run can consolidate both without duplicate filenames.
p = 'scripts/make_reviewer2_curated_figures.m'
replace_once(
    p,
    'export_shared_result_legend(out,"orbit_family_selection_legend", ...\n        ["NHO";"SHO";"NNRHO";"SNRHO";"DRO"],lines(5),style, ...\n        \'Kind\',\'patch\',\'NumColumns\',3);\n    manifest = add_manifest(manifest,"comparison","orbit_family_selection_legend", ...\n',
    'export_shared_result_legend(out,"comparison_orbit_family_selection_legend", ...\n        ["NHO";"SHO";"NNRHO";"SNRHO";"DRO"],lines(5),style, ...\n        \'Kind\',\'patch\',\'NumColumns\',3);\n    manifest = add_manifest(manifest,"comparison","comparison_orbit_family_selection_legend", ...\n',
)
replace_once(
    p,
    'export_shared_result_legend(out,"orbit_family_selection_legend", ...\n        ["NHO";"SHO";"NNRHO";"SNRHO";"DRO"],lines(5),style, ...\n        \'Kind\',\'patch\',\'NumColumns\',3);\n    manifest = add_manifest(manifest,"baseline","orbit_family_selection_legend", ...\n',
    'export_shared_result_legend(out,"baseline_orbit_family_selection_legend", ...\n        ["NHO";"SHO";"NNRHO";"SNRHO";"DRO"],lines(5),style, ...\n        \'Kind\',\'patch\',\'NumColumns\',3);\n    manifest = add_manifest(manifest,"baseline","baseline_orbit_family_selection_legend", ...\n',
)

# Keep the legacy direct curated-runtime path consistent with the normal final
# renderer: one shared legend strip and no embedded panel legend.
replace_once(
    p,
    '    r = reports.runtime;\n    out = prepare_output(r.analysisDirectory,saveFigures);\n',
    '    r = reports.runtime;\n    out = prepare_output(r.analysisDirectory,saveFigures);\n'
    '    runtimeOptimizers = style.optimizerOrder(ismember(style.optimizerOrder,string(r.runtimeResults.Optimizer)));\n'
    '    runtimeColors = colors_for_optimizers(runtimeOptimizers,style);\n'
    '    export_shared_result_legend(out,"runtime_1200_legend", ...\n'
    '        [optimizer_labels(runtimeOptimizers);"6000-FE GA reference"], ...\n'
    '        [runtimeColors;0.30 0.30 0.30],style, ...\n'
    '        \'LineStyles\',[repmat("-",numel(runtimeOptimizers),1);"--"],\'NumColumns\',2);\n'
    '    manifest = add_manifest(manifest,"runtime","runtime_1200_legend", ...\n'
    '        "Shared optimizer/reference legend for the 1200-FE subfigure group.");\n',
)
replace_once(
    p,
    "lgd = legend(ax,legendHandles,cellstr(legendLabels),'Location','northoutside', ...\n    'Orientation','horizontal','NumColumns',min(numel(legendLabels),6),'Box','off');\nstyle_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);\n",
    "export_figure(fig,out,stem,saveFigures,style);\n",
)

# Ensure the dedicated runtime renderer compares string types explicitly and
# documents the new shared-legend convention.
p = 'scripts/make_reviewer2_runtime_figures.m'
replace_once(
    p,
    'runtimeOptimizers = style.optimizerOrder(ismember(style.optimizerOrder,r.runtimeResults.Optimizer));\n',
    'runtimeOptimizers = style.optimizerOrder(ismember(style.optimizerOrder,string(r.runtimeResults.Optimizer)));\n',
)
replace_once(
    p,
    '% The optimizer identity is already explicit on the x-axis of the two bar\n% charts, so those charts intentionally do not create one legend entry per\n% optimizer. MATLAB\'s bar() returns one Bar object for this flat-colored\n% categorical chart; pairing that single handle with five optimizer labels\n% produces the "Ignoring extra legend entries" warning. The objective chart\n% therefore uses a legend only for the dashed 6000-FE GA reference.\n% The convergence chart uses one graphics handle per optimizer curve and\n',
    '% The optimizer identity is explicit on the x-axis of the two bar charts,\n% while a single legend-only EPS supplies the common optimizer/reference key\n% for the complete 1200-FE subfigure group. Individual panels contain no\n% embedded legends, matching the trajectory and Monte Carlo grid convention.\n% The convergence chart uses one graphics handle per optimizer curve and\n',
)

# Return the serial/parallel legend EPS so run_manuscript_figures consolidates
# it into MANUSCRIPT_OUTPUT/figures/ with the two data panels.
p = 'scripts/plot_parallel_speed.m'
replace_once(p, 'style=reviewer2_paper_style(); files=strings(2,1);\n',
             'style=reviewer2_paper_style(); files=strings(3,1);\n')
replace_once(
    p,
    'export_shared_result_legend(outputDirectory,"parallel_speed_lg_legend", ...\n    ["Serial";"Parallel"],style.optimizerColors(1:2,:),style, ...\n    \'LineStyles\',["-";"--"],\'NumColumns\',2);\n',
    'legendFiles=export_shared_result_legend(outputDirectory,"parallel_speed_lg_legend", ...\n    ["Serial";"Parallel"],style.optimizerColors(1:2,:),style, ...\n    \'LineStyles\',["-";"--"],\'NumColumns\',2);\nfiles(3)=legendFiles(1);\n',
)

print('Finalized common-legend implementation.')
