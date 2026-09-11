function format_manuscript_legend(ax,lgd,style,~)
%FORMAT_MANUSCRIPT_LEGEND Apply the shared manuscript legend/layout rules.
%
% All layout values come from reviewer2_paper_style. Two-dimensional legends
% use one common centered row above the common 2-D axes rectangle. Three-
% dimensional legends are also centered, use the slightly larger 3-D legend
% font, and sit close to the trajectories by overlapping only the otherwise
% unused upper portion of the centered 3-D axes box.

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

% Prefer a single row. Only introduce additional rows when the complete row
% does not fit the common manuscript canvas.
lgd.Orientation = 'horizontal';
columns = min(numel(labels),style.legendMaxColumns);
lgd.NumColumns = max(columns,1);
lgd.Units = 'normalized';
drawnow;
while lgd.Position(3) > style.legendMaxWidth && columns > 1
    columns = columns-1;
    lgd.NumColumns = columns;
    drawnow;
end

% If a very long label still exceeds the standard canvas, reduce the legend
% font only as a last resort. The configured 2-D/3-D sizes are otherwise kept.
while lgd.Position(3) > style.legendMaxWidth && ...
        lgd.FontSize > style.legendMinFontSize
    lgd.FontSize = lgd.FontSize-1;
    drawnow;
end

% Position is controlled explicitly rather than by MATLAB's northoutside
% heuristic. Setting Location to none prevents MATLAB from shifting the legend
% relative to the axes after we center it on the common export canvas.
pos = lgd.Position;
lgd.Location = 'none';
pos(1) = 0.5-pos(3)/2;
pos(2) = plotPosition(2)+plotPosition(4)+legendGap;
pos(2) = min(pos(2),0.985-pos(4));
lgd.Position = pos;
lgd.AutoUpdate = 'off';

% Restore the central axes rectangle after legend construction. This makes
% every 2-D plot share the same top/bottom margins and every 3-D plot share the
% same horizontal center regardless of the order in which MATLAB created it.
ax.Units = 'normalized';
ax.PositionConstraint = 'innerposition';
ax.Position = plotPosition;
drawnow;

format_manuscript_ticks(ax);
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
