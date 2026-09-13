from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected one match in {path}, found {n}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1))


def insert_after_once(path, marker, addition):
    replace_once(path, marker, marker + addition)

# -----------------------------------------------------------------------------
# Shared style: legend-only EPS strips and enlarged legend-free result axes.
# -----------------------------------------------------------------------------
replace_once(
    'scripts/reviewer2_paper_style.m',
    "style.sharedGeometryLegendWidth = 3.10;\nstyle.sharedGeometryLegendHeight = 1.55;\n",
    "style.sharedGeometryLegendWidth = 3.10;\nstyle.sharedGeometryLegendHeight = 1.55;\nstyle.sharedResultLegendWidth = 3.10;\nstyle.sharedResultLegendHeight = 1.55;\n",
)
replace_once(
    'scripts/reviewer2_paper_style.m',
    "style.metricPlotPosition = [0.15 0.18 0.70 0.53];\n",
    "style.metricPlotPosition = [0.15 0.18 0.70 0.53];\nstyle.metricPlotPositionNoLegend = [0.15 0.16 0.70 0.70];\n",
)

# -----------------------------------------------------------------------------
# Curated results: export one common legend EPS per logical multi-panel group,
# remove repeated legends from every panel, and reclaim the panel's top space.
# -----------------------------------------------------------------------------
p = 'scripts/make_reviewer2_curated_figures.m'

insert_after_once(
    p,
    "    refs = matched_baselines(baselineResults,missions,3,1,'BestJMean','BestJStd');\n",
    "    compOptimizers = string(r.optimizers);\n"
    "    compColors = colors_for_optimizers(compOptimizers,style);\n"
    "    export_shared_result_legend(out,\"comparison_6000_metrics_legend\", ...\n"
    "        [optimizer_labels(compOptimizers);\"GA baseline\"], ...\n"
    "        [compColors;0.30 0.30 0.30],style, ...\n"
    "        'LineStyles',[repmat(\"-\",numel(compOptimizers),1);\"--\"],'NumColumns',2);\n"
    "    export_shared_result_legend(out,\"comparison_6000_convergence_legend\", ...\n"
    "        optimizer_labels(compOptimizers),compColors,style,'NumColumns',2);\n"
    "    manifest = add_manifest(manifest,\"comparison\",\"comparison_6000_metrics_legend\", ...\n"
    "        \"Shared GA/PSO/ABC/ACO and GA-baseline legend for the 6000-FE metric grid.\");\n"
    "    manifest = add_manifest(manifest,\"comparison\",\"comparison_6000_convergence_legend\", ...\n"
    "        \"Shared optimizer legend for the 6000-FE convergence grid.\");\n",
)

insert_after_once(
    p,
    "    familyData = build_comparison_family_data(r);\n",
    "    export_shared_result_legend(out,\"orbit_family_selection_legend\", ...\n"
    "        [\"NHO\";\"SHO\";\"NNRHO\";\"SNRHO\";\"DRO\"],lines(5),style, ...\n"
    "        'Kind','patch','NumColumns',3);\n"
    "    manifest = add_manifest(manifest,\"comparison\",\"orbit_family_selection_legend\", ...\n"
    "        \"Shared five-family legend for orbit-family selection panels.\");\n",
)

insert_after_once(
    p,
    "    missions = [\"LUNAR_GATEWAY\",\"LOW_THRUST_TRANSFER\",\"GATEWAY_IMPULSE\"];\n\n    observerSpecs = { ...\n",
    "    export_shared_result_legend(out,\"baseline_observer_metric_legend\", ...\n"
    "        [\"Angles only\";\"Angles + range\"],style.measurementColors,style,'NumColumns',2);\n"
    "    export_shared_result_legend(out,\"baseline_observer_convergence_legend\", ...\n"
    "        [\"3 observers\";\"5 observers\";\"7 observers\";\"10 observers\"], ...\n"
    "        lines(4),style,'NumColumns',2);\n"
    "    export_shared_result_legend(out,\"baseline_duration_metric_legend\", ...\n"
    "        [\"3 observers\";\"5 observers\";\"7 observers\";\"10 observers\"], ...\n"
    "        lines(4),style,'NumColumns',2);\n"
    "    export_shared_result_legend(out,\"baseline_duration_convergence_legend\", ...\n"
    "        [\"1 period\";\"3 periods\";\"5 periods\"],lines(3),style,'NumColumns',3);\n"
    "    manifest = add_manifest(manifest,\"baseline\",\"baseline_observer_metric_legend\", ...\n"
    "        \"Shared AO/AR legend for observer-count metric panels.\");\n"
    "    manifest = add_manifest(manifest,\"baseline\",\"baseline_observer_convergence_legend\", ...\n"
    "        \"Shared observer-count legend for AO/AR convergence panels.\");\n"
    "    manifest = add_manifest(manifest,\"baseline\",\"baseline_duration_metric_legend\", ...\n"
    "        \"Shared observer-count legend for propagation-duration metric panels.\");\n"
    "    manifest = add_manifest(manifest,\"baseline\",\"baseline_duration_convergence_legend\", ...\n"
    "        \"Shared propagation-duration legend for AO/AR convergence panels.\");\n\n    observerSpecs = { ...\n",
)

insert_after_once(
    p,
    "    familyData = build_baseline_family_data(r);\n",
    "    export_shared_result_legend(out,\"orbit_family_selection_legend\", ...\n"
    "        [\"NHO\";\"SHO\";\"NNRHO\";\"SNRHO\";\"DRO\"],lines(5),style, ...\n"
    "        'Kind','patch','NumColumns',3);\n"
    "    manifest = add_manifest(manifest,\"baseline\",\"orbit_family_selection_legend\", ...\n"
    "        \"Shared five-family legend for orbit-family selection panels.\");\n",
)

# Objective/screening section has the same mission marker as baseline, so use a
# more specific context marker.
insert_after_once(
    p,
    "    out = prepare_output(r.analysisDirectory,saveFigures);\n    missions = [\"LUNAR_GATEWAY\",\"LOW_THRUST_TRANSFER\",\"GATEWAY_IMPULSE\"];\n\n    % Screening ON/OFF: only the physical metrics requested for the paper.\n",
    "    export_shared_result_legend(out,\"ga_screening_metric_legend\", ...\n"
    "        [\"Screening ON\";\"Screening OFF\"],style.configurationColors(1:2,:),style,'NumColumns',2);\n"
    "    export_shared_result_legend(out,\"ga_screening_convergence_legend\", ...\n"
    "        [\"Screening ON\";\"Screening OFF\"],style.configurationColors(1:2,:),style,'NumColumns',2);\n"
    "    export_shared_result_legend(out,\"ga_objective_orbit_family_selection_legend\", ...\n"
    "        [\"NHO\";\"SHO\";\"NNRHO\";\"SNRHO\";\"DRO\"],lines(5),style, ...\n"
    "        'Kind','patch','NumColumns',3);\n"
    "    manifest = add_manifest(manifest,\"objective_screening\",\"ga_screening_metric_legend\", ...\n"
    "        \"Shared screening ON/OFF legend for metric panels.\");\n"
    "    manifest = add_manifest(manifest,\"objective_screening\",\"ga_screening_convergence_legend\", ...\n"
    "        \"Shared screening ON/OFF legend for convergence panels.\");\n"
    "    manifest = add_manifest(manifest,\"objective_screening\",\"ga_objective_orbit_family_selection_legend\", ...\n"
    "        \"Shared five-family legend for objective-component family panels.\");\n\n    % Screening ON/OFF: only the physical metrics requested for the paper.\n",
)

# Remove per-panel legends from comparison metrics.
replace_once(
    p,
    "lgd = legend(ax,legendHandles,cellstr(legendLabels),'Location','northoutside', ...\n    'Orientation','horizontal','NumColumns',min(numel(legendLabels),5),'Box','off');\nstyle_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);\n",
    "export_figure(fig,out,stem,saveFigures,style);\n",
)

# Remove AO/AR legend from baseline observer metrics.
replace_once(
    p,
    "lgd = legend(ax,handles,{'Angles only','Angles + range'}, ...\n    'Location','northoutside','Orientation','horizontal','Box','off');\nstyle_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);\n",
    "export_figure(fig,out,stem,saveFigures,style);\n",
)

# Remove observer-count legend from duration metrics.
replace_once(
    p,
    "lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal','NumColumns',2,'Box','off');\nstyle_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);\n",
    "export_figure(fig,out,stem,saveFigures,style);\n",
)

# Remove screening metric legend.
replace_once(
    p,
    "lgd = legend(ax,b,{'Screening ON','Screening OFF'},'Location','northoutside', ...\n    'Orientation','horizontal','Box','off'); style_legend(lgd,ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
    "export_figure(fig,out,stem,saveFigures,style);\n",
)

# Remove both orbit-family legends.
replace_once(
    p,
    "lgd = legend(ax,b,cellstr(families),'Location','northoutside', ...\n    'Orientation','horizontal','NumColumns',5,'Box','off');\nstyle_legend(lgd,ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
    "export_figure(fig,out,stem,saveFigures,style);\n",
)
replace_once(
    p,
    "style_axes(ax,style); lgd = legend(ax,b,cellstr(families),'Location','northoutside', ...\n    'Orientation','horizontal','NumColumns',5,'Box','off'); style_legend(lgd,ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
    "style_axes(ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
)

# All curated convergence panels are part of common-legend subfigure groups.
replace_once(
    p,
    "style_axes(ax,style); lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...\n    'NumColumns',min(numel(handles),5),'Box','off'); style_legend(lgd,ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
    "style_axes(ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
)

# Use the larger legend-free axes box for all curated result panels.
replace_once(
    p,
    "set(ax,'Units','normalized','Position',style.metricPlotPosition);\n",
    "set(ax,'Units','normalized','Position',style.metricPlotPositionNoLegend);\n",
)

# -----------------------------------------------------------------------------
# Focused 1200-FE renderer: one six-entry common legend, no panel legends.
# -----------------------------------------------------------------------------
p = 'scripts/make_reviewer2_runtime_figures.m'
insert_after_once(
    p,
    "baseline = matched_baseline(baselineResults,\"LUNAR_GATEWAY\",3,1);\n",
    "runtimeOptimizers = style.optimizerOrder(ismember(style.optimizerOrder,r.runtimeResults.Optimizer));\n"
    "runtimeColors = colors_for_optimizers(runtimeOptimizers,style);\n"
    "export_shared_result_legend(out,\"runtime_1200_legend\", ...\n"
    "    [optimizer_labels(runtimeOptimizers);\"6000-FE GA reference\"], ...\n"
    "    [runtimeColors;0.30 0.30 0.30],style, ...\n"
    "    'LineStyles',[repmat(\"-\",numel(runtimeOptimizers),1);\"--\"],'NumColumns',2);\n",
)

replace_once(
    p,
    "    hBase = plot(ax,[0.55 height(R)+0.45],[baseline.Mean baseline.Mean],'--', ...\n        'Color',[0.30 0.30 0.30],'LineWidth',1.5, ...\n        'DisplayName','6000-FE GA reference');\n    lgd = legend(ax,hBase,{'6000-FE GA reference'},'Location','northoutside', ...\n        'Orientation','horizontal','Box','off');\n    style_legend(lgd,ax,style);\n",
    "    plot(ax,[0.55 height(R)+0.45],[baseline.Mean baseline.Mean],'--', ...\n        'Color',[0.30 0.30 0.30],'LineWidth',1.5, ...\n        'HandleVisibility','off');\n",
)
replace_once(
    p,
    "style_axes(ax,style);\nlgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...\n    'NumColumns',min(numel(handles),5),'Box','off');\nstyle_legend(lgd,ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
    "style_axes(ax,style);\nexport_figure(fig,out,stem,saveFigures,style);\n",
)
replace_once(
    p,
    "set(ax,'Units','normalized','Position',style.metricPlotPosition);\n",
    "set(ax,'Units','normalized','Position',style.metricPlotPositionNoLegend);\n",
)
replace_once(
    p,
    "manifest = table( ...\n    repmat(\"runtime\",3,1), ...\n    [\"runtime_1200_objective\";\"runtime_1200_runtime\";\"runtime_1200_convergence\"], ...\n    [\"Equal-1200-FE mean final-best objective with matched 6000-FE GA reference.\"; ...\n     \"Equal-1200-FE mean computational cost showing BO scaling penalty.\"; ...\n     \"Five-method equal-FE mean convergence comparison.\"], ...\n",
    "manifest = table( ...\n    repmat(\"runtime\",4,1), ...\n    [\"runtime_1200_legend\";\"runtime_1200_objective\";\"runtime_1200_runtime\";\"runtime_1200_convergence\"], ...\n    [\"Shared optimizer/reference legend for the 1200-FE subfigure group.\"; ...\n     \"Equal-1200-FE mean final-best objective with matched 6000-FE GA reference.\"; ...\n     \"Equal-1200-FE mean computational cost showing BO scaling penalty.\"; ...\n     \"Five-method equal-FE mean convergence comparison.\"], ...\n",
)

# -----------------------------------------------------------------------------
# Serial/parallel benchmark: two legend-free panels plus one common legend EPS.
# -----------------------------------------------------------------------------
p = 'scripts/plot_parallel_speed.m'
replace_once(
    p,
    "    ax=axes(fig,'Units','normalized','Position',style.metricPlotPosition); hold(ax,'on');\n",
    "    ax=axes(fig,'Units','normalized','Position',style.metricPlotPositionNoLegend); hold(ax,'on');\n",
)
replace_once(
    p,
    "    lgd=legend(ax,'Location','northoutside','Orientation','horizontal','Box','off');\n    format_manuscript_legend(ax,lgd,style,style.metricPlotPosition);\n",
    "",
)
insert_after_once(
    p,
    "end\n% Keep the numeric printout beside the final figures as well as in the raw run.\n",
    "export_shared_result_legend(outputDirectory,\"parallel_speed_lg_legend\", ...\n"
    "    [\"Serial\";\"Parallel\"],style.optimizerColors(1:2,:),style, ...\n"
    "    'LineStyles',[\"-\";\"--\"],'NumColumns',2);\n"
    "% Keep the numeric printout beside the final figures as well as in the raw run.\n",
)
# -----------------------------------------------------------------------------
# Static manuscript-figure audit.
# -----------------------------------------------------------------------------
p = 'tests/test_reviewer2_paper_figures_configuration.m'
insert_after_once(
    p,
    "legendColumnsText = string(fileread(fullfile(projectDir,'scripts','manuscript_legend_columns.m')));\n",
    "sharedResultLegendText = string(fileread(fullfile(projectDir,'scripts','export_shared_result_legend.m')));\n",
)
replace_once(
    p,
    "    \"style.sharedGeometryLegendHeight = 1.55\", ...\n",
    "    \"style.sharedGeometryLegendHeight = 1.55\", ...\n    \"style.sharedResultLegendWidth = 3.10\", ...\n    \"style.sharedResultLegendHeight = 1.55\", ...\n",
)
replace_once(
    p,
    "    \"style.metricPlotPosition = [0.15 0.18 0.70 0.53]\", ...\n",
    "    \"style.metricPlotPosition = [0.15 0.18 0.70 0.53]\", ...\n    \"style.metricPlotPositionNoLegend = [0.15 0.16 0.70 0.70]\", ...\n",
)
replace_once(
    p,
    "assert(contains(parallelText,'style.figureWidth style.figureHeight') && ...\n    contains(parallelText,'format_manuscript_legend'), ...\n    'Parallel figures must use the common canvas and legend formatter.');\n",
    "assert(contains(parallelText,'style.figureWidth style.figureHeight') && ...\n    contains(parallelText,'export_shared_result_legend') && ...\n    contains(parallelText,'style.metricPlotPositionNoLegend'), ...\n    'Parallel panels must use a separate common legend and the legend-free axes box.');\n",
)
insert_after_once(
    p,
    "    'Dense manuscript grids must export compact, readable shared legends with up to three rows.');\n",
    "assert(contains(sharedResultLegendText,'style.sharedResultLegendWidth') && ...\n"
    "    contains(sharedResultLegendText,'style.sharedLegendFontSize') && ...\n"
    "    contains(curatedText,'comparison_6000_metrics_legend') && ...\n"
    "    contains(curatedText,'comparison_6000_convergence_legend') && ...\n"
    "    contains(curatedText,'baseline_observer_metric_legend') && ...\n"
    "    contains(curatedText,'baseline_observer_convergence_legend') && ...\n"
    "    contains(curatedText,'baseline_duration_metric_legend') && ...\n"
    "    contains(curatedText,'baseline_duration_convergence_legend') && ...\n"
    "    contains(curatedText,'ga_screening_metric_legend') && ...\n"
    "    contains(curatedText,'ga_screening_convergence_legend') && ...\n"
    "    contains(curatedText,'ga_objective_orbit_family_selection_legend') && ...\n"
    "    contains(curatedText,'orbit_family_selection_legend'), ...\n"
    "    'All repeated result legends must be exported as separate common EPS strips.');\n",
)

print('Separated common result legends from multi-panel manuscript figures.')
