from pathlib import Path

p = Path('scripts/plot_study_definition_figures.m')
text = p.read_text()

required = [
    'sharedFamilyLegendFiles = export_shared_result_legend(',
    '"orbit_family_trajectory_legend"',
    'legendHandle = gobjects(0);',
    'outputs.sharedLegend = sharedFamilyLegendFiles(1);',
]
for marker in required:
    if marker not in text:
        raise RuntimeError(f'Missing expected shared-family-legend marker: {marker}')

old = """function finalize_centered_3d_axes(ax,legendHandle,plotPosition)\n%FINALIZE_CENTERED_3D_AXES Match the final paper trajectory construction.\n\nstyle = reviewer2_paper_style();\naxis(ax,'tight');\nxlim(ax,pad_axis_limits(ax.XLim,style.geometryXPadding));\nylim(ax,pad_axis_limits(ax.YLim,style.geometryYPadding));\nzlim(ax,pad_axis_limits(ax.ZLim,style.geometryZPadding));\naxis(ax,'vis3d');\n\nformat_manuscript_legend(ax,legendHandle,style,plotPosition);\n\nend\n"""
new = """function finalize_centered_3d_axes(ax,legendHandle,plotPosition)\n%FINALIZE_CENTERED_3D_AXES Match the final paper trajectory construction.\n\nstyle = reviewer2_paper_style();\naxis(ax,'tight');\nxlim(ax,pad_axis_limits(ax.XLim,style.geometryXPadding));\nylim(ax,pad_axis_limits(ax.YLim,style.geometryYPadding));\nzlim(ax,pad_axis_limits(ax.ZLim,style.geometryZPadding));\naxis(ax,'vis3d');\n\nif isempty(legendHandle) || ~all(isgraphics(legendHandle))\n    % Shared-legend panels intentionally have no local legend. Preserve the\n    % requested larger plot rectangle; the final export pass centers the axes.\n    ax.Units = 'normalized';\n    ax.PositionConstraint = 'innerposition';\n    ax.Position = plotPosition;\n    drawnow;\nelse\n    format_manuscript_legend(ax,legendHandle,style,plotPosition);\nend\n\nend\n"""
if text.count(old) != 1:
    raise RuntimeError(f'Expected one finalize_centered_3d_axes block, found {text.count(old)}')
text = text.replace(old,new,1)
p.write_text(text)
print('Validated shared family legend and fixed legend-free 3-D finalization.')
