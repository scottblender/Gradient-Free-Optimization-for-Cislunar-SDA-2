function format_manuscript_legend(ax,lgd,style,~)
%FORMAT_MANUSCRIPT_LEGEND Apply the shared manuscript legend/layout rules.
%
% All layout values come from reviewer2_paper_style. Legends with three or
% more entries are arranged in two centered rows by default. Two-dimensional
% and three-dimensional legends use the same wrapping rule, while 3-D plots
% retain the slightly larger legend font and tighter trajectory spacing.

labels = string(lgd.String);
labels = contextual_legend_labels(labels);
labels = abbreviate_manuscript_text(labels);
lgd.String = cellstr(labels);
lgd.FontName = style.fontName;
lgd.FontWeight = style.fontWeight;
lgd.Box = 'off';

% MATLAB reports [0 90] for ordinary 2-D axes. Perspective trajectory axes
% use another elevation; top-down DRO plots remain 2-D for layout purposes.
is3D = abs(ax.View(2)-90) > 1e-8;
if is3D
    plotPosition = style.geometryPlotPosition;
    lgd.FontSize = style.geometryLegendFontSize;
    lgd.ItemTokenSize = style.geometryLegendItemTokenSize;
    legendGap = style.geometryLegendGap;
else
    plotPosition = style.metricPlotPosition;
    lgd.FontSize = style.legendFontSize;
    lgd.ItemTokenSize = style.legendItemTokenSize;
    legendGap = style.legend2DGap;
end

% Use two centered rows whenever there are at least three legend entries.
% Examples: 4 entries -> 2 columns x 2 rows; 8 entries -> 4 columns x 2 rows.
lgd.Orientation = 'horizontal';
columns = manuscript_legend_columns(labels,style);
lgd.NumColumns = columns;
lgd.Units = 'normalized';
drawnow;

% Preserve the requested two-row layout while possible. If a very long
% legend still exceeds the common canvas, reduce the font first. Only if the
% minimum font still does not fit do we allow an additional wrapped row.
while lgd.Position(3) > style.legendMaxWidth && ...
        lgd.FontSize > style.legendMinFontSize
    lgd.FontSize = lgd.FontSize-1;
    drawnow;
end
while lgd.Position(3) > style.legendMaxWidth && columns > 1
    columns = columns-1;
    lgd.NumColumns = columns;
    drawnow;
end

% Position is controlled explicitly rather than by MATLAB's northoutside
% heuristic. Setting Location to none prevents MATLAB from shifting the legend
% relative to the axes after we place it on the common export canvas.
pos = lgd.Position;
lgd.Location = 'none';
pos(1) = 0.5-pos(3)/2;
pos(2) = plotPosition(2)+plotPosition(4)+legendGap;
pos(2) = min(pos(2),0.985-pos(4));
lgd.Position = pos;
lgd.AutoUpdate = 'off';

% Restore the shared axes rectangle after legend construction.
ax.Units = 'normalized';
ax.PositionConstraint = 'innerposition';
ax.Position = plotPosition;
drawnow;

% Finalize tick density before measuring TightInset so the centering pass uses
% the exact tick-label geometry that will be exported.
format_manuscript_ticks(ax);
drawnow;

% One final construction-time pass is used for BOTH 2-D and 3-D plots. The
% legend is centered horizontally on the figure bounding box; the axes are
% shifted to equalize the complete left/right visible margins; then the axes
% and legend move together vertically so the complete top/bottom visible
% margins are equal while the configured legend-to-plot spacing is preserved.
fig = ancestor(ax,'figure');
center_manuscript_content(fig,ax,lgd);
drawnow;
end


function labels = contextual_legend_labels(labels)
%CONTEXTUAL_LEGEND_LABELS Use compact mission-specific trajectory labels.
labels = string(labels);

hasEndpoints = any(labels == "Endpoint orbits");
hasTargetTrajectory = any(labels == "Target trajectory");

% Result-geometry legends.
if hasTargetTrajectory
    if hasEndpoints
        labels(labels == "Target trajectory") = "LT";
    elseif any(labels == "Nominal Gateway")
        labels(labels == "Target trajectory") = "GI";
        labels(labels == "Nominal Gateway") = "Nominal LG";
    else
        labels(labels == "Target trajectory") = "LG";
    end
end

% Study-definition trajectory legends.
if any(labels == "Transfer")
    labels(labels == "Transfer") = "LT";
end
if any(labels == "Post-impulse")
    labels(labels == "Post-impulse") = "GI";
    labels(labels == "Nominal Gateway") = "Nominal LG";
elseif any(labels == "Nominal Gateway") && ~hasTargetTrajectory
    labels(labels == "Nominal Gateway") = "LG";
end

% Keep the high-frequency trajectory legend entries very short so the full
% row remains centered and close to the plot at manuscript scale.
labels(labels == "Observer orbits") = "Obs.";
labels(labels == "Observers") = "Obs.";
labels(labels == "Endpoint orbits") = "Endpt.";
labels(labels == "Endpoints") = "Endpt.";
labels(labels == "Candidate slots") = "Slots";
labels(labels == "Excluded endpoint") = "Endpt.";
labels(labels == "Endpoint") = "Endpt.";

% Preserve explicit equilibrium-point names in orbit-database legends. The
% family labels L1/L2 and the physical "L1 point"/"L2 point" markers must not
% be conflated.
end
