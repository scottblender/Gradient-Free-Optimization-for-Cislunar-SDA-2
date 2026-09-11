function format_manuscript_legend(ax,lgd,style,plotPosition)
%FORMAT_MANUSCRIPT_LEGEND Finalize a horizontal manuscript legend.
% Formatting occurs during plot construction only. Start with one row and
% add rows only when the legend would exceed the available figure width.
% The axes then expand upward to the legend so unused vertical whitespace
% does not make the plotted content artificially small in the EPS panel.

labels = abbreviate_manuscript_text(string(lgd.String));
lgd.String = cellstr(labels);
lgd.FontName = style.fontName;
lgd.FontSize = style.fontSize;
lgd.FontWeight = 'bold';
lgd.Box = 'off';
lgd.Orientation = 'horizontal';
lgd.Location = 'northoutside';
lgd.Units = 'normalized';

columns = min(numel(labels),style.legendMaxColumns);
lgd.NumColumns = max(columns,1);
drawnow;
while lgd.Position(3) > 0.92 && columns > 1
    columns = columns-1;
    lgd.NumColumns = columns;
    drawnow;
end

% Keep the legend near the top of the paper, then use all available space
% below it for the axes. Previously the axes retained their shorter nominal
% height, leaving a large blank band between the plot and legend.
lp = lgd.Position;
lp(1) = max(0.02,0.5-lp(3)/2);
lp(2) = 0.97-lp(4);
lgd.Position = lp;
lgd.AutoUpdate = 'off';

plotPosition(4) = lp(2)-style.geometryLegendGap-plotPosition(2);
assert(plotPosition(4) > 0.30 && lp(1) >= 0,'Manuscript:LegendSpace', ...
    'Legend text is too large for this canvas; shorten labels in the plotter.');
ax.Units = 'normalized';
ax.PositionConstraint = 'innerposition';
ax.Position = plotPosition;
drawnow;

% Remove only the crowded central zero from symmetric three-tick 3-D axes.
% Data limits and two-dimensional zero baselines are unchanged.
format_manuscript_ticks(ax);
end
